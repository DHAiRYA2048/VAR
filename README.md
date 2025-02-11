# High Level Overview
![Overview](https://github.com/user-attachments/assets/2dba3ebb-52d6-4fb5-be5b-a23edd53ceb6)

# Main Idea
![next_scale_pred](https://github.com/user-attachments/assets/2e64d241-7e0d-4392-aca0-3f5383284008)

# Steps

## 1. Image tokenization (VQVAE)

1.1 Input Image  

1.2 Encoder: Extract laten features from the image  

1.3 Quantization:  
  - Apply a convolution (`quant_conv`) to the latent feature map.  
  - Use `VectorQuantizer2` to:  
    • Downsample the feature map at multiple scales (as defined by `v_patch_nums`).  
    • For each scale, compute nearest codebook vector.  
    • Aggregate quantized outputs with residual refinement (via one of: non‑shared, fully shared, or partially shared `φ` layers).  

1.4 Output: Multi‑scale discrete tokens (and reconstruction loss computed via MSE)  

## 2. Prepare transformer inputs

2.1 Teacher‑Forcing Setup:  
  - Extract ground-truth token indices from VQVAE output (via `f_to_idxBl_or_fhat`).  

2.2 Conditioning:  
  - Obtain class label → class embedding (with SOS token).  

2.3 Positional & Level Embeddings:  
  - Add “pos_start”, absolute positional embeddings (`pos_1LC`), and level embeddings (`lvl_embed`) to indicate token scale.  

2.4 Word Embedding:  
  - Project VQVAE token embeddings via a linear layer (`word_embed`) into the transformer’s embedding space.  

2.5 Result: A concatenated token sequence: [SOS tokens + teacher-forced tokens]  

## 3. Attention mask

• For training, construct an attention mask that prevents future tokens from being attended to.  

• This mask ensures autoregressive (left-to-right) behavior.  

## 4. Transfomer input embedding 

4.1 Input: Teacher-forcing token sequence (from Step 2).  

4.2 Embedding integration:  
  - Combine word embeddings, class embedding, positional, and level embeddings.  

4.3 Output: A full input sequence (shape: [B, L_total, C]).  

## 5. Processing through transformer

5.1 For each of N stacked `AdaLNSelfAttn` blocks:  

5.1.1 Adaptive Layer Normalization (`AdaLN`):  
   - Normalizes input tokens.  
   - Applies learned conditional scaling & shifting (derived from class embedding via `SharedAdaLin`).  

5.1.2 Multi-Head Self-Attention:  
   - Linearly projects inputs into Q, K, V.  
   - Computes attention scores (optionally using flash/memory‑efficient implementations).  
   - Applies (masked) softmax & aggregates values.  

5.1.3 Feed‑Forward Network (FFN):  
   - Two linear layers with GELU activation & dropout.  
   - Residual connections add FFN output to input.  

5.2 Output: Updated token representations capturing spatial and semantic dependencies.  

## 6. Output & Loss

6.1 Final Adaptive LN: Apply `AdaLNBeforeHead` to condition output on the class embedding.  

6.2 Linear Projection: Map tokens to logits over the VQVAE codebook (vocab size).  

6.3 Loss Computation:  
  - Compute cross‑entropy loss (with label smoothing) between predicted logits and ground-truth token indices.  

## 7. Inference

7.1 Initialization:  
  - Begin with SOS token block (from class embedding).  

7.2 For each scale in the token pyramid:  

7.2.1 Transformer Processing:  
     - Process current token sequence through all `AdaLNSelfAttn` blocks (as in Step 5) with no causal mask (using KV caching).  

 7.2.2 Guided Sampling:  
     - Adjust logits using classifier‑free guidance (`CFG`):  
       * Scale logits based on current generation stage.  
     - Sample token indices via top‑k/top‑p methods.  

7.3 Convert sampled indices into token embeddings (via VQVAE codebook lookup).  


## 8. Generation at different scales 

8.1 Update Latent Representation (`f_hat`):  
  - Use VQVAE’s get_next_autoregressive_input to “add” the new token embeddings into `f_hat`.  

8.2 Upsampling:  
  - Interpolate `f_hat` to match the next scale’s resolution (defined by `patch_nums`).  

8.3 Prepare Next Token Map:  
  - Re-embed the updated `f_hat` (via `word_embed`) and add corresponding positional/level embeddings.  

## 9. Final image reconstruction via VQVAE  

9.1 After processing all scales, the final latent `f_hat` represents the full-resolution token map.  

9.2 Pass `f_hat` through post‑processing conv and the VQVAE decoder to reconstruct the high‑resolution image.  

9.3 De‑normalize output to [0, 1].  

## 10. Training vs. Inference

• Training:  
  - Teacher forcing is used to supply ground‑truth tokens, and cross‑entropy loss guides the transformer.  

• Inference:  
  - The transformer generates tokens autoregressively using classifier‑free guidance and sampling, then feeds the tokens progressively into higher scales.


# Glossary
* quant_conv: A convolutional layer applied to the latent feature map before quantization.
* VectorQuantizer2: The module that quantizes latent features by mapping them to the nearest codebook vectors at multiple scales.
* v_patch_nums: A tuple defining the number of patches (or scales) used for quantization (e.g., (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)).
* φ layers: Convolutional layers used for residual refinement during quantization; they can be non‑shared, fully shared, or partially shared.
* f_to_idxBl_or_fhat: A function that converts latent feature maps into either token indices or a reconstructed latent representation.
* f_hat: The accumulated latent representation after integrating quantized outputs across scales.
* class_emb: The embedding layer that projects class labels into the transformer’s embedding space for conditioning.
* SOS token: The start‑of‑sequence token generated from the class embedding, marking the beginning of a token sequence.
* pos_start: A learned positional embedding for the initial token block.
* pos_1LC: Absolute positional embeddings for the full token sequence.
* lvl_embed: Level embeddings that differentiate tokens from different scales.
* word_embed: A linear layer that projects VQVAE token embeddings into the transformer’s embedding space.
* AdaLNSelfAttn: A transformer block that combines Adaptive Layer Normalization (AdaLN), multi‑head self‑attention, and a Feed‑Forward Network.
* SharedAdaLin: A conditional linear layer that provides scaling and shifting parameters derived from conditioning signals.
* flash_attn_func: An optimized attention function from the flash‑attn library for faster computation.
* memory_efficient_attention: An optimized attention function from xformers that reduces memory usage.
* AdaLNBeforeHead: A final adaptive layer normalization module applied before the output linear head.
* CFG (Classifier‑Free Guidance): A technique to adjust logits during inference to steer generation by mixing conditional and unconditional predictions.
* top‑k/top‑p sampling: Techniques to sample token indices from logits by restricting the sampling pool to the top‑k most likely tokens or the smallest set of tokens whose cumulative probability exceeds top‑p.
* patch_nums: Similar to v_patch_nums, defines the resolutions for each stage in progressive generation.
* post_quant_conv: A convolutional layer applied after quantization to refine the latent before decoding.
