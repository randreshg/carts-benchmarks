/* Test file for isolated Transformer neural network functions */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// -----------------------------------------------------------------------------
// Transformer model structures (copied from llama2.c)

typedef struct {
  int dim;        // transformer dimension
  int hidden_dim; // for ffn layers
  int n_layers;   // number of layers
  int n_heads;    // number of query heads
  int n_kv_heads; // number of key/value heads (can be < query heads because of
                  // multiquery)
  int vocab_size; // vocabulary size, usually 256 (byte-level)
  int seq_len;    // max sequence length
} Config;

typedef struct {
  // token embedding table
  float *token_embedding_table; // (vocab_size, dim)
  // weights for rmsnorms
  float *rms_att_weight; // (layer, dim) rmsnorm weights
  float *rms_ffn_weight; // (layer, dim)
  // weights for matmuls. note dim == n_heads * head_size
  float *wq; // (layer, dim, n_heads * head_size)
  float *wk; // (layer, dim, n_kv_heads * head_size)
  float *wv; // (layer, dim, n_kv_heads * head_size)
  float *wo; // (layer, n_heads * head_size, dim)
  // weights for ffn
  float *w1; // (layer, hidden_dim, dim)
  float *w2; // (layer, dim, hidden_dim)
  float *w3; // (layer, hidden_dim, dim)
  // final rmsnorm
  float *rms_final_weight; // (dim,)
  // (optional) classifier weights for the logits, on the last layer
  float *wcls;
} TransformerWeights;

typedef struct {
  // current wave of activations
  float *x;      // activation at current time stamp (dim,)
  float *xb;     // same, but inside a residual branch (dim,)
  float *xb2;    // an additional buffer just for convenience (dim,)
  float *hb;     // buffer for hidden dimension in the ffn (hidden_dim,)
  float *hb2;    // buffer for hidden dimension in the ffn (hidden_dim,)
  float *q;      // query (dim,)
  float *k;      // key (dim,)
  float *v;      // value (dim,)
  float *att;    // buffer for scores/attention values (n_heads, seq_len)
  float *logits; // output logits
  // kv cache
  float *key_cache;   // (layer, seq_len, dim)
  float *value_cache; // (layer, seq_len, dim)
} RunState;

typedef struct {
  Config config; // the hyperparameters of the architecture (the blueprint)
  TransformerWeights weights; // the weights of the model
  RunState state; // buffers for the "wave" of activations in the forward pass
  // some more state needed to properly clean up the memory mapping (sigh)
  int fd;            // file descriptor for memory mapping
  float *data;       // memory mapped data pointer
  ssize_t file_size; // size of the checkpoint file in bytes
} Transformer;

// -----------------------------------------------------------------------------
// Neural network blocks; the dynamics of the Transformer

void rmsnorm(float *o, float *x, float *weight, int size) {
  // calculate sum of squares
  float ss = 0.0f;
  for (int j = 0; j < size; j++) {
    ss += x[j] * x[j];
  }
  ss /= size;
  ss += 1e-5f;
  ss = 1.0f / sqrtf(ss);
  // normalize and scale
  for (int j = 0; j < size; j++) {
    o[j] = weight[j] * (ss * x[j]);
  }
}

void softmax(float *x, int size) {
  // find max value (for numerical stability)
  float max_val = x[0];
  for (int i = 1; i < size; i++) {
    if (x[i] > max_val) {
      max_val = x[i];
    }
  }
  // exp and sum
  float sum = 0.0f;
  for (int i = 0; i < size; i++) {
    x[i] = expf(x[i] - max_val);
    sum += x[i];
  }
  // normalize
  for (int i = 0; i < size; i++) {
    x[i] /= sum;
  }
}

void matmul(float *xout, float *x, float *w, int n, int d) {
  // W (d,n) @ x (n,) -> xout (d,)
  // by far the most amount of time is spent inside this little function
  for (int i = 0; i < d; i++) {
    float val = 0.0f;
    for (int j = 0; j < n; j++) {
      val += w[i * n + j] * x[j];
    }
    xout[i] = val;
  }
}

float *forward(Transformer *transformer, int token, int pos) {
  // a few convenience variables
  Config *p = &transformer->config;
  TransformerWeights *w = &transformer->weights;
  RunState *s = &transformer->state;
  float *x = s->x;
  int dim = p->dim;
  int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
  int kv_mul =
      p->n_heads /
      p->n_kv_heads; // integer multiplier of the kv sharing in multiquery
  int hidden_dim = p->hidden_dim;
  int head_size = dim / p->n_heads;

  // copy the token embedding into x
  float *content_row = w->token_embedding_table + token * dim;
  memcpy(x, content_row, dim * sizeof(*x));

  // forward all the layers
  for (int l = 0; l < (int)p->n_layers; l++) {

    // attention rmsnorm
    rmsnorm(s->xb, x, w->rms_att_weight + l * dim, dim);

    // key and value point to the kv cache
    int loff = l * p->seq_len * kv_dim; // kv cache layer offset for convenience
    s->k = s->key_cache + loff + pos * kv_dim;
    s->v = s->value_cache + loff + pos * kv_dim;

    // qkv matmuls for this position
    matmul(s->q, s->xb, w->wq + l * dim * dim, dim, dim);
    matmul(s->k, s->xb, w->wk + l * dim * kv_dim, dim, kv_dim);
    matmul(s->v, s->xb, w->wv + l * dim * kv_dim, dim, kv_dim);

    // RoPE relative positional encoding: complex-valued rotate q and k in each
    // head
    for (int i = 0; i < dim; i += 2) {
      int head_dim = i % head_size;
      float freq = 1.0f / powf(10000.0f, head_dim / (float)head_size);
      float val = pos * freq;
      float fcr = cosf(val);
      float fci = sinf(val);
      int rotn = i < kv_dim ? 2 : 1; // how many vectors? 2 = q & k, 1 = q only
      for (int v = 0; v < rotn; v++) {
        float *vec =
            v == 0 ? s->q : s->k; // the vector to rotate (query or key)
        float v0 = vec[i];
        float v1 = vec[i + 1];
        vec[i] = v0 * fcr - v1 * fci;
        vec[i + 1] = v0 * fci + v1 * fcr;
      }
    }

    // multihead attention. iterate over all heads
    int h;
#pragma omp parallel for private(h)
    for (h = 0; h < p->n_heads; h++) {
      // get the query vector for this head
      float *q = s->q + h * head_size;
      // attention scores for this head
      float *att = s->att + h * p->seq_len;
      // iterate over all timesteps, including the current one
      for (int t = 0; t <= pos; t++) {
        // get the key vector for this head and at this timestep
        float *k = s->key_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
        // calculate the attention score as the dot product of q and k
        float score = 0.0f;
        for (int i = 0; i < head_size; i++) {
          score += q[i] * k[i];
        }
        score /= sqrtf(head_size);
        // save the score to the attention buffer
        att[t] = score;
      }

      // softmax the scores to get attention weights, from 0..pos inclusively
      softmax(att, pos + 1);

      // weighted sum of the values, store back into xb
      float *xb = s->xb + h * head_size;
      memset(xb, 0, head_size * sizeof(float));
      for (int t = 0; t <= pos; t++) {
        // get the value vector for this head and at this timestep
        float *v =
            s->value_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
        // get the attention weight for this timestep
        float a = att[t];
        // accumulate the weighted value into xb
        for (int i = 0; i < head_size; i++) {
          xb[i] += a * v[i];
        }
      }
    }

    // final matmul to get the output of the attention
    matmul(s->xb2, s->xb, w->wo + l * dim * dim, dim, dim);

    // residual connection back into x
    for (int i = 0; i < dim; i++) {
      x[i] += s->xb2[i];
    }

    // ffn rmsnorm
    rmsnorm(s->xb, x, w->rms_ffn_weight + l * dim, dim);

    // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
    // first calculate self.w1(x) and self.w3(x)
    matmul(s->hb, s->xb, w->w1 + l * dim * hidden_dim, dim, hidden_dim);
    matmul(s->hb2, s->xb, w->w3 + l * dim * hidden_dim, dim, hidden_dim);

    // SwiGLU non-linearity
    for (int i = 0; i < hidden_dim; i++) {
      float val = s->hb[i];
      // silu(x)=x*σ(x), where σ(x) is the logistic sigmoid
      val *= (1.0f / (1.0f + expf(-val)));
      // elementwise multiply with w3(x)
      val *= s->hb2[i];
      s->hb[i] = val;
    }

    // final matmul to get the output of the ffn
    matmul(s->xb, s->hb, w->w2 + l * dim * hidden_dim, hidden_dim, dim);

    // residual connection
    for (int i = 0; i < dim; i++) {
      x[i] += s->xb[i];
    }
  }

  // final rmsnorm
  rmsnorm(x, x, w->rms_final_weight, dim);

  // classifier into logits
  matmul(s->logits, x, w->wcls, p->dim, p->vocab_size);
  return s->logits;
}

// -----------------------------------------------------------------------------
// Memory allocation functions (simplified versions)

void malloc_run_state(RunState *s, Config *p) {
  // we calloc instead of malloc to keep valgrind happy
  int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
  s->x = (float *)calloc(p->dim, sizeof(float));
  s->xb = (float *)calloc(p->dim, sizeof(float));
  s->xb2 = (float *)calloc(p->dim, sizeof(float));
  s->hb = (float *)calloc(p->hidden_dim, sizeof(float));
  s->hb2 = (float *)calloc(p->hidden_dim, sizeof(float));
  s->q = (float *)calloc(p->dim, sizeof(float));
  s->key_cache =
      (float *)calloc(p->n_layers * p->seq_len * kv_dim, sizeof(float));
  s->value_cache =
      (float *)calloc(p->n_layers * p->seq_len * kv_dim, sizeof(float));
  s->att = (float *)calloc(p->n_heads * p->seq_len, sizeof(float));
  s->logits = (float *)calloc(p->vocab_size, sizeof(float));
  // ensure all mallocs went fine
  if (!s->x || !s->xb || !s->xb2 || !s->hb || !s->hb2 || !s->q ||
      !s->key_cache || !s->value_cache || !s->att || !s->logits) {
    fprintf(stderr, "malloc failed!\n");
    exit(EXIT_FAILURE);
  }
}

void free_run_state(RunState *s) {
  free(s->x);
  free(s->xb);
  free(s->xb2);
  free(s->hb);
  free(s->hb2);
  free(s->q);
  free(s->att);
  free(s->logits);
  free(s->key_cache);
  free(s->value_cache);
}

// -----------------------------------------------------------------------------
// Test driver

void initialize_test_data(Transformer *transformer) {
  Config *p = &transformer->config;
  TransformerWeights *w = &transformer->weights;

  // Initialize with small test values
  // For a real test, you'd load actual weights from a checkpoint file

  // Initialize token embeddings with small random-like values
  for (int i = 0; i < p->vocab_size * p->dim; i++) {
    w->token_embedding_table[i] = 0.01f * (float)(rand() % 100 - 50) / 50.0f;
  }

  // Initialize weights for all layers
  for (int l = 0; l < p->n_layers; l++) {
    // RMS norm weights (attention)
    for (int i = 0; i < p->dim; i++) {
      w->rms_att_weight[l * p->dim + i] = 1.0f;
      w->rms_ffn_weight[l * p->dim + i] = 1.0f;
    }

    // Attention weights - initialize to small values
    for (int i = 0; i < p->dim * p->dim; i++) {
      w->wq[l * p->dim * p->dim + i] =
          0.01f * (float)(rand() % 100 - 50) / 50.0f;
      w->wk[l * p->dim * p->dim + i] =
          0.01f * (float)(rand() % 100 - 50) / 50.0f;
      w->wv[l * p->dim * p->dim + i] =
          0.01f * (float)(rand() % 100 - 50) / 50.0f;
      w->wo[l * p->dim * p->dim + i] =
          0.01f * (float)(rand() % 100 - 50) / 50.0f;
    }

    // FFN weights
    for (int i = 0; i < p->dim * p->hidden_dim; i++) {
      w->w1[l * p->dim * p->hidden_dim + i] =
          0.01f * (float)(rand() % 100 - 50) / 50.0f;
      w->w3[l * p->dim * p->hidden_dim + i] =
          0.01f * (float)(rand() % 100 - 50) / 50.0f;
    }
    for (int i = 0; i < p->hidden_dim * p->dim; i++) {
      w->w2[l * p->hidden_dim * p->dim + i] =
          0.01f * (float)(rand() % 100 - 50) / 50.0f;
    }
  }

  // Final RMS norm weights
  for (int i = 0; i < p->dim; i++) {
    w->rms_final_weight[i] = 1.0f;
  }

  // Classifier weights (using token embeddings as in shared weights)
  w->wcls = w->token_embedding_table;
}

int main() {
  printf("Testing isolated Transformer neural network functions\n");

  // Initialize a small test configuration
  Transformer transformer;
  Config *p = &transformer.config;

  // Small test configuration
  p->dim = 64;         // Small embedding dimension
  p->hidden_dim = 256; // Small hidden dimension for FFN
  p->n_layers = 2;     // Just 2 layers for testing
  p->n_heads = 4;      // 4 attention heads
  p->n_kv_heads = 4;   // Same as n_heads (no multiquery)
  p->vocab_size = 256; // Small vocabulary
  p->seq_len = 32;     // Short sequence length

  printf("Configuration: dim=%d, hidden_dim=%d, n_layers=%d, n_heads=%d, "
         "vocab_size=%d\n",
         p->dim, p->hidden_dim, p->n_layers, p->n_heads, p->vocab_size);

  // Allocate memory for transformer
  TransformerWeights *w = &transformer.weights;

  // Allocate weight arrays
  w->token_embedding_table =
      (float *)malloc(p->vocab_size * p->dim * sizeof(float));
  w->rms_att_weight = (float *)malloc(p->n_layers * p->dim * sizeof(float));
  w->rms_ffn_weight = (float *)malloc(p->n_layers * p->dim * sizeof(float));
  w->wq = (float *)malloc(p->n_layers * p->dim * p->dim * sizeof(float));
  w->wk = (float *)malloc(p->n_layers * p->dim * p->dim * sizeof(float));
  w->wv = (float *)malloc(p->n_layers * p->dim * p->dim * sizeof(float));
  w->wo = (float *)malloc(p->n_layers * p->dim * p->dim * sizeof(float));
  w->w1 = (float *)malloc(p->n_layers * p->dim * p->hidden_dim * sizeof(float));
  w->w2 = (float *)malloc(p->n_layers * p->hidden_dim * p->dim * sizeof(float));
  w->w3 = (float *)malloc(p->n_layers * p->dim * p->hidden_dim * sizeof(float));
  w->rms_final_weight = (float *)malloc(p->dim * sizeof(float));

  // Allocate run state
  malloc_run_state(&transformer.state, p);

  // Initialize with test data
  srand(42); // Fixed seed for reproducible tests
  initialize_test_data(&transformer);

  // Test forward pass
  printf("Testing forward pass...\n");

  int test_token = 42; // Test with token 42
  int test_pos = 0;    // Position 0

  float *logits = forward(&transformer, test_token, test_pos);

  printf("Forward pass completed. First 10 logits: ");
  for (int i = 0; i < 10; i++) {
    printf("%.4f ", logits[i]);
  }
  printf("\n");

  // Test individual functions
  printf("\nTesting individual functions...\n");

  // Test rmsnorm
  float test_x[4] = {1.0f, 2.0f, 3.0f, 4.0f};
  float test_weight[4] = {1.0f, 1.0f, 1.0f, 1.0f};
  float test_o[4];
  rmsnorm(test_o, test_x, test_weight, 4);
  printf("RMSNorm test: [%.4f, %.4f, %.4f, %.4f] -> [%.4f, %.4f, %.4f, %.4f]\n",
         test_x[0], test_x[1], test_x[2], test_x[3], test_o[0], test_o[1],
         test_o[2], test_o[3]);

  // Test softmax
  float test_softmax[4] = {1.0f, 2.0f, 3.0f, 4.0f};
  softmax(test_softmax, 4);
  printf("Softmax test: [1.0, 2.0, 3.0, 4.0] -> [%.4f, %.4f, %.4f, %.4f]\n",
         test_softmax[0], test_softmax[1], test_softmax[2], test_softmax[3]);

  // Test matmul (small 2x3 matrix @ 3x1 vector)
  float test_w[6] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}; // 2x3 matrix
  float test_vec[3] = {1.0f, 1.0f, 1.0f};                 // 3x1 vector
  float test_result[2];
  matmul(test_result, test_vec, test_w, 3, 2); // n=3, d=2
  printf("Matmul test: [1,2,3; 4,5,6] @ [1,1,1] = [%.1f, %.1f]\n",
         test_result[0], test_result[1]);

  // Cleanup
  free_run_state(&transformer.state);

  free(w->token_embedding_table);
  free(w->rms_att_weight);
  free(w->rms_ffn_weight);
  free(w->wq);
  free(w->wk);
  free(w->wv);
  free(w->wo);
  free(w->w1);
  free(w->w2);
  free(w->w3);
  free(w->rms_final_weight);

  printf("All tests completed successfully!\n");
  return 0;
}
