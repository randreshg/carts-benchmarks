/* Transformer model implementation for carts benchmarks */

#include "arts/Utils/Benchmarks/CartsBenchmarks.h"
#include <float.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

// Size configuration - can be overridden via compiler flags
#ifndef DIM
#define DIM 64
#endif
#ifndef HIDDEN_DIM
#define HIDDEN_DIM 256
#endif
#ifndef N_LAYERS
#define N_LAYERS 2
#endif
#ifndef N_HEADS
#define N_HEADS 4
#endif
#ifndef N_KV_HEADS
#define N_KV_HEADS 4
#endif
#ifndef VOCAB_SIZE
#define VOCAB_SIZE 256
#endif
#ifndef SEQ_LEN
#define SEQ_LEN 32
#endif

#define KV_DIM ((DIM * N_KV_HEADS) / N_HEADS)
#define KV_MUL (N_HEADS / N_KV_HEADS)
#define HEAD_SIZE (DIM / N_HEADS)

// ============================================================================
// Memory allocation helpers
// ============================================================================

static float **alloc_2d(int rows, int cols) {
  float **arr = (float **)malloc(rows * sizeof(float *));
  for (int i = 0; i < rows; i++) {
    arr[i] = (float *)malloc(cols * sizeof(float));
  }
  return arr;
}

static float ***alloc_3d(int d1, int d2, int d3) {
  float ***arr = (float ***)malloc(d1 * sizeof(float **));
  for (int i = 0; i < d1; i++) {
    arr[i] = (float **)malloc(d2 * sizeof(float *));
    for (int j = 0; j < d2; j++) {
      arr[i][j] = (float *)malloc(d3 * sizeof(float));
    }
  }
  return arr;
}

static void free_2d(float **arr, int rows) {
  for (int i = 0; i < rows; i++) {
    free(arr[i]);
  }
  free(arr);
}

static void free_3d(float ***arr, int d1, int d2) {
  for (int i = 0; i < d1; i++) {
    for (int j = 0; j < d2; j++) {
      free(arr[i][j]);
    }
    free(arr[i]);
  }
  free(arr);
}

// ============================================================================
// Helper functions
// ============================================================================

static void zero_floats(float *buffer, int count) {
  for (int i = 0; i < count; i++) {
    buffer[i] = 0.0f;
  }
}

static void rmsnorm(float *o, float *x_vec, float *weight, int size) {
  float ss = 0.0f;
  for (int j = 0; j < size; j++) {
    ss += x_vec[j] * x_vec[j];
  }
  ss /= size;
  ss += 1e-5f;
  ss = 1.0f / sqrtf(ss);
#pragma omp parallel for
  for (int j = 0; j < size; j++) {
    o[j] = weight[j] * (ss * x_vec[j]);
  }
}

static void softmax(float *x_vec, int size) {
  float max_val = -FLT_MAX;
  for (int i = 0; i < size; i++) {
    if (x_vec[i] > max_val) {
      max_val = x_vec[i];
    }
  }
  float sum = 0.0f;
  for (int i = 0; i < size; i++) {
    x_vec[i] = expf(x_vec[i] - max_val);
    sum += x_vec[i];
  }
  float inv_sum = 1.0f / sum;
  for (int i = 0; i < size; i++) {
    x_vec[i] *= inv_sum;
  }
}

static void matmul(float *xout, float *x_vec, float **w, int n, int d) {
  // W (d,n) @ x (n,) -> xout (d,)
  // by far the most amount of time is spent inside this little function
  int i;
#pragma omp parallel for schedule(static) private(i)
  for (i = 0; i < d; i++) {
    float val = 0.0f;
    for (int j = 0; j < n; j++) {
      val += w[i][j] * x_vec[j];
    }
    xout[i] = val;
  }
}

// ============================================================================
// Initialization functions
// ============================================================================

static void initialize_state(float *x, float *xb, float *xb2, float *hb,
                             float *hb2, float *q_buf, float **att_buf,
                             float *logits, float ***key_cache,
                             float ***value_cache) {
  zero_floats(x, DIM);
  zero_floats(xb, DIM);
  zero_floats(xb2, DIM);
  zero_floats(hb, HIDDEN_DIM);
  zero_floats(hb2, HIDDEN_DIM);
  zero_floats(q_buf, DIM);
  for (int h = 0; h < N_HEADS; h++) {
    zero_floats(att_buf[h], SEQ_LEN);
  }
  zero_floats(logits, VOCAB_SIZE);
  for (int l = 0; l < N_LAYERS; l++) {
    for (int s = 0; s < SEQ_LEN; s++) {
      zero_floats(key_cache[l][s], KV_DIM);
      zero_floats(value_cache[l][s], KV_DIM);
    }
  }
}

static void initialize_test_data(float **token_embedding_table,
                                 float **rms_att_weight, float **rms_ffn_weight,
                                 float ***wq, float ***wk, float ***wv,
                                 float ***wo, float ***w1, float ***w2,
                                 float ***w3, float *rms_final_weight) {
  // Token embedding table [VOCAB_SIZE][DIM]
  for (int i = 0; i < VOCAB_SIZE; i++) {
    for (int j = 0; j < DIM; j++) {
      token_embedding_table[i][j] = 0.01f * (float)(rand() % 100 - 50) / 50.0f;
    }
  }

  // Per-layer weights
  for (int l = 0; l < N_LAYERS; l++) {
    // RMS weights [N_LAYERS][DIM]
    for (int i = 0; i < DIM; i++) {
      rms_att_weight[l][i] = 1.0f;
      rms_ffn_weight[l][i] = 1.0f;
    }

    // wq, wo [N_LAYERS][DIM][DIM]
    for (int i = 0; i < DIM; i++) {
      for (int j = 0; j < DIM; j++) {
        wq[l][i][j] = 0.01f * (float)(rand() % 100 - 50) / 50.0f;
        wo[l][i][j] = 0.01f * (float)(rand() % 100 - 50) / 50.0f;
      }
    }

    // wk, wv [N_LAYERS][KV_DIM][DIM]
    for (int i = 0; i < KV_DIM; i++) {
      for (int j = 0; j < DIM; j++) {
        wk[l][i][j] = 0.01f * (float)(rand() % 100 - 50) / 50.0f;
        wv[l][i][j] = 0.01f * (float)(rand() % 100 - 50) / 50.0f;
      }
    }

    // w1, w3 [N_LAYERS][HIDDEN_DIM][DIM]
    for (int i = 0; i < HIDDEN_DIM; i++) {
      for (int j = 0; j < DIM; j++) {
        w1[l][i][j] = 0.01f * (float)(rand() % 100 - 50) / 50.0f;
        w3[l][i][j] = 0.01f * (float)(rand() % 100 - 50) / 50.0f;
      }
    }

    // w2 [N_LAYERS][DIM][HIDDEN_DIM]
    for (int i = 0; i < DIM; i++) {
      for (int j = 0; j < HIDDEN_DIM; j++) {
        w2[l][i][j] = 0.01f * (float)(rand() % 100 - 50) / 50.0f;
      }
    }
  }

  // Final RMS weight [DIM]
  for (int i = 0; i < DIM; i++) {
    rms_final_weight[i] = 1.0f;
  }
}

// ============================================================================
// Transformer forward pass
// ============================================================================

static float *forward(
    // Model weights
    float **token_embedding_table, float **rms_att_weight,
    float **rms_ffn_weight, float ***wq, float ***wk, float ***wv, float ***wo,
    float ***w1, float ***w2, float ***w3, float *rms_final_weight,
    // State buffers
    float *x, float *xb, float *xb2, float *hb, float *hb2, float *q_buf,
    float **att_buf, float *logits, float ***key_cache, float ***value_cache,
    // Position info
    int token, int pos) {

  // Copy the token embedding into x
#pragma omp parallel for
  for (int i = 0; i < DIM; i++) {
    x[i] = token_embedding_table[token][i];
  }

  // Forward all the layers
  for (int l = 0; l < N_LAYERS; l++) {
    // Attention rmsnorm
    rmsnorm(xb, x, rms_att_weight[l], DIM);

    // QKV matmuls for this position
    matmul(q_buf, xb, wq[l], DIM, DIM);
    matmul(key_cache[l][pos], xb, wk[l], DIM, KV_DIM);
    matmul(value_cache[l][pos], xb, wv[l], DIM, KV_DIM);

    // RoPE relative positional encoding: complex-valued rotate q and k in each
    // head
#pragma omp parallel for
    for (int i = 0; i < DIM; i += 2) {
      int head_dim = i % HEAD_SIZE;
      float freq = 1.0f / powf(10000.0f, (float)head_dim / (float)HEAD_SIZE);
      float angle = pos * freq;
      float fcr = cosf(angle);
      float fci = sinf(angle);
      float v0 = q_buf[i];
      float v1 = q_buf[i + 1];
      q_buf[i] = v0 * fcr - v1 * fci;
      q_buf[i + 1] = v0 * fci + v1 * fcr;
      if (i < KV_DIM) {
        float *kvec = key_cache[l][pos];
        float k0 = kvec[i];
        float k1 = kvec[i + 1];
        kvec[i] = k0 * fcr - k1 * fci;
        kvec[i + 1] = k0 * fci + k1 * fcr;
      }
    }

    // Multihead attention - iterate over all heads
    int h;
#pragma omp parallel for schedule(static) private(h)
    for (h = 0; h < N_HEADS; h++) {
      // Get the query vector for this head
      float *head_q = q_buf + h * HEAD_SIZE;
      // Attention scores for this head
      float *head_att = att_buf[h];
      // Iterate over all timesteps, including the current one
      for (int t = 0; t <= pos; t++) {
        // Get the key vector for this head and at this timestep
        int kv_head = h / KV_MUL;
        float *head_k = key_cache[l][t] + kv_head * HEAD_SIZE;
        // Calculate the attention score as the dot product of q and k
        float score = 0.0f;
        for (int i = 0; i < HEAD_SIZE; i++) {
          score += head_q[i] * head_k[i];
        }
        score /= sqrtf((float)HEAD_SIZE);
        // Save the score to the attention buffer
        head_att[t] = score;
      }

      // Softmax the scores to get attention weights, from 0..pos inclusively
      softmax(head_att, pos + 1);

      // Weighted sum of the values, store back into xb
      float *xb_head = xb + h * HEAD_SIZE;
      for (int i = 0; i < HEAD_SIZE; i++) {
        xb_head[i] = 0.0f;
      }
      for (int t = 0; t <= pos; t++) {
        // Get the value vector for this head and at this timestep
        int kv_head = h / KV_MUL;
        float *head_v = value_cache[l][t] + kv_head * HEAD_SIZE;
        // Get the attention weight for this timestep
        float weight = head_att[t];
        // Accumulate the weighted value into xb
        for (int i = 0; i < HEAD_SIZE; i++) {
          xb_head[i] += weight * head_v[i];
        }
      }
    }

    // Final matmul to get the output of the attention
    matmul(xb2, xb, wo[l], DIM, DIM);

    // Residual connection back into x
#pragma omp parallel for
    for (int i = 0; i < DIM; i++) {
      x[i] += xb2[i];
    }

    // FFN rmsnorm
    rmsnorm(xb, x, rms_ffn_weight[l], DIM);

    // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
    // First calculate self.w1(x) and self.w3(x)
    matmul(hb, xb, w1[l], DIM, HIDDEN_DIM);
    matmul(hb2, xb, w3[l], DIM, HIDDEN_DIM);

    // SwiGLU non-linearity
#pragma omp parallel for
    for (int i = 0; i < HIDDEN_DIM; i++) {
      float val = hb[i];
      // silu(x)=x*sigma(x), where sigma(x) is the logistic sigmoid
      val *= (1.0f / (1.0f + expf(-val)));
      // Elementwise multiply with w3(x)
      val *= hb2[i];
      hb[i] = val;
    }

    // Final matmul to get the output of the ffn
    matmul(xb, hb, w2[l], HIDDEN_DIM, DIM);

    // Residual connection
#pragma omp parallel for
    for (int i = 0; i < DIM; i++) {
      x[i] += xb[i];
    }
  }

  // Final rmsnorm
  rmsnorm(x, x, rms_final_weight, DIM);

  // Classifier into logits
  matmul(logits, x, token_embedding_table, DIM, VOCAB_SIZE);

  return logits;
}

// ============================================================================
// Main
// ============================================================================

int main(void) {
  // Pre-warm OMP thread pool for fair comparison (must be first)
  CARTS_BENCHMARKS_START();
  CARTS_E2E_TIMER_START("transformer");

  printf("Testing isolated Transformer neural network functions\n");
  printf("Configuration: dim=%d, hidden_dim=%d, n_layers=%d, n_heads=%d, "
         "vocab_size=%d\n",
         DIM, HIDDEN_DIM, N_LAYERS, N_HEADS, VOCAB_SIZE);

  // Allocate model weights using pointer-to-pointer pattern
  float **token_embedding_table = alloc_2d(VOCAB_SIZE, DIM);
  float **rms_att_weight = alloc_2d(N_LAYERS, DIM);
  float **rms_ffn_weight = alloc_2d(N_LAYERS, DIM);
  float ***wq = alloc_3d(N_LAYERS, DIM, DIM);
  float ***wk = alloc_3d(N_LAYERS, KV_DIM, DIM);
  float ***wv = alloc_3d(N_LAYERS, KV_DIM, DIM);
  float ***wo = alloc_3d(N_LAYERS, DIM, DIM);
  // FFN weights: w1, w3 are [HIDDEN_DIM x DIM], w2 is [DIM x HIDDEN_DIM]
  // matmul(out, in, W, n, d) computes out[d] = W[d][n] @ in[n]
  float ***w1 = alloc_3d(N_LAYERS, HIDDEN_DIM, DIM);
  float ***w2 = alloc_3d(N_LAYERS, DIM, HIDDEN_DIM);
  float ***w3 = alloc_3d(N_LAYERS, HIDDEN_DIM, DIM);
  float *rms_final_weight = (float *)malloc(DIM * sizeof(float));

  // Allocate state buffers
  float *x = (float *)malloc(DIM * sizeof(float));
  float *xb = (float *)malloc(DIM * sizeof(float));
  float *xb2 = (float *)malloc(DIM * sizeof(float));
  float *hb = (float *)malloc(HIDDEN_DIM * sizeof(float));
  float *hb2 = (float *)malloc(HIDDEN_DIM * sizeof(float));
  float *q_buf = (float *)malloc(DIM * sizeof(float));
  float **att_buf = alloc_2d(N_HEADS, SEQ_LEN);
  float *logits = (float *)malloc(VOCAB_SIZE * sizeof(float));
  float ***key_cache = alloc_3d(N_LAYERS, SEQ_LEN, KV_DIM);
  float ***value_cache = alloc_3d(N_LAYERS, SEQ_LEN, KV_DIM);

  // Initialize
  initialize_state(x, xb, xb2, hb, hb2, q_buf, att_buf, logits, key_cache,
                   value_cache);
  srand(42);
  initialize_test_data(token_embedding_table, rms_att_weight, rms_ffn_weight,
                       wq, wk, wv, wo, w1, w2, w3, rms_final_weight);

  printf("Testing forward pass...\n");
  int test_token = 42;
  int test_pos = 0;

  // CARTS_KERNEL_TIMER_START("transformer");
  float *logits_out =
      forward(token_embedding_table, rms_att_weight, rms_ffn_weight, wq, wk, wv,
              wo, w1, w2, w3, rms_final_weight, x, xb, xb2, hb, hb2, q_buf,
              att_buf, logits, key_cache, value_cache, test_token, test_pos);
  // CARTS_KERNEL_TIMER_STOP("transformer");

  printf("Forward pass completed. First 10 logits: ");
  for (int i = 0; i < 10; i++) {
    printf("%.4f ", logits_out[i]);
  }
  printf("\n");

  // Verification
  float checksum = 0.0f;
  for (int i = 0; i < VOCAB_SIZE; i++) {
    checksum += logits_out[i];
  }
  CARTS_BENCH_CHECKSUM(checksum);

  printf("\nTesting individual functions...\n");

  float test_x[4] = {1.0f, 2.0f, 3.0f, 4.0f};
  float test_weight[4] = {1.0f, 1.0f, 1.0f, 1.0f};
  float test_o[4];
  rmsnorm(test_o, test_x, test_weight, 4);
  printf("RMSNorm test: [%.4f, %.4f, %.4f, %.4f] -> [%.4f, %.4f, %.4f, %.4f]\n",
         test_x[0], test_x[1], test_x[2], test_x[3], test_o[0], test_o[1],
         test_o[2], test_o[3]);

  float test_softmax[4] = {1.0f, 2.0f, 3.0f, 4.0f};
  softmax(test_softmax, 4);
  printf("Softmax test: [1.0, 2.0, 3.0, 4.0] -> [%.4f, %.4f, %.4f, %.4f]\n",
         test_softmax[0], test_softmax[1], test_softmax[2], test_softmax[3]);

  // Matmul test with 2D array
  float **test_w = alloc_2d(2, 3);
  test_w[0][0] = 1.0f;
  test_w[0][1] = 2.0f;
  test_w[0][2] = 3.0f;
  test_w[1][0] = 4.0f;
  test_w[1][1] = 5.0f;
  test_w[1][2] = 6.0f;
  float test_vec[3] = {1.0f, 1.0f, 1.0f};
  float test_result[2];
  matmul(test_result, test_vec, test_w, 3, 2);
  printf("Matmul test: [1,2,3; 4,5,6] @ [1,1,1] = [%.1f, %.1f]\n",
         test_result[0], test_result[1]);
  free_2d(test_w, 2);

  printf("All tests completed successfully!\n");

  // Free model weights
  free_2d(token_embedding_table, VOCAB_SIZE);
  free_2d(rms_att_weight, N_LAYERS);
  free_2d(rms_ffn_weight, N_LAYERS);
  free_3d(wq, N_LAYERS, DIM);
  free_3d(wk, N_LAYERS, KV_DIM);
  free_3d(wv, N_LAYERS, KV_DIM);
  free_3d(wo, N_LAYERS, DIM);
  free_3d(w1, N_LAYERS, HIDDEN_DIM);
  free_3d(w2, N_LAYERS, DIM);
  free_3d(w3, N_LAYERS, HIDDEN_DIM);
  free(rms_final_weight);

  // Free state buffers
  free(x);
  free(xb);
  free(xb2);
  free(hb);
  free(hb2);
  free(q_buf);
  free_2d(att_buf, N_HEADS);
  free(logits);
  free_3d(key_cache, N_LAYERS, SEQ_LEN);
  free_3d(value_cache, N_LAYERS, SEQ_LEN);

  CARTS_E2E_TIMER_STOP();
  CARTS_BENCHMARKS_STOP();

  return 0;
}
