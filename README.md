# High Level Overview
![Overview](https://github.com/user-attachments/assets/2dba3ebb-52d6-4fb5-be5b-a23edd53ceb6)

# Steps

## 1. Image Tokenization via VQVAE

1.1 Input Image  

1.2 Encoder: Extract latent features from the image  

1.3 Quantization:  
  - Apply a convolution (quant_conv) to the latent feature map.  
  - Use `VectorQuantizer2` to:  
    • Downsample the feature map at multiple scales (as defined by `v_patch_nums`).  
    • For each scale, compute nearest codebook vector (with optional L2‑normalization).  
    • Aggregate quantized outputs with residual refinement (via one of: non‑shared, fully shared, or partially shared `φ` layers).  

1.4 Output: Multi‑scale discrete tokens (and reconstruction loss computed via MSE)  

## 2. Preparing Transformer Inputs

2.1 Teacher‑Forcing Setup:  
  - Extract ground-truth token indices from VQVAE output (via `f_to_idxBl_or_fhat`).  

2.2 Conditioning:  
  - Obtain class label → class embedding (with SOS token).  

2.3 Positional & Level Embeddings:  
  - Add “pos_start”, absolute positional embeddings (`pos_1LC`), and level embeddings (`lvl_embed`) to indicate token scale.  

2.4 Word Embedding:  
  - Project VQVAE token embeddings via a linear layer (word_embed) into the transformer’s embedding space.  

2.5 Result: A concatenated token sequence: [SOS tokens + teacher-forced tokens]  

## 3. Causal Attention Mask Setup

• For training, construct an attention mask that prevents future tokens from being attended to.  

• This mask ensures autoregressive (left-to-right) behavior.  

## 4. Transformer Input Embedding 

4.1 Input: Teacher-forcing token sequence (from Step 2).  

4.2 Embedding Integration:  
  - Combine word embeddings, class embedding, positional, and level embeddings.  

4.3 Output: A full input sequence (shape: [B, L_total, C]).  

## 5. Processing Through Transformer Backbone

5.1 For each of N stacked AdaLNSelfAttn blocks:  

5.1.1 Adaptive Layer Normalization (AdaLN):  
   - Normalizes input tokens.  
   - Applies learned conditional scaling & shifting (derived from class embedding via SharedAdaLin).  

5.1.2 Multi-Head Self-Attention:  
   - Linearly projects inputs into Q, K, V.  
   - Computes attention scores (optionally using flash/memory‑efficient implementations).  
   - Applies (masked) softmax & aggregates values.  

5.1.3 Feed‑Forward Network (FFN):  
   - Two linear layers with GELU activation & dropout.  
   - Residual connections add FFN output to input.  

5.2 Output: Updated token representations capturing spatial and semantic dependencies.  

## 6. Output Projection & Loss Computation

6.1 Final Adaptive LN: Apply AdaLNBeforeHead to condition output on the class embedding.  

6.2 Linear Projection: Map tokens to logits over the VQVAE codebook (vocab size).  

6.3 Loss Computation (Training Mode):  
  - Compute cross‑entropy loss (with label smoothing) between predicted logits and ground-truth token indices.  

## 7. Autoregressive Generation (Inference Mode)

7.1 Initialization:  
  - Begin with SOS token block (from class embedding).  

7.2 For each scale in the token pyramid:  

7.2.1 Transformer Processing:  
     - Process current token sequence through all AdaLNSelfAttn blocks (as in Step 5) with no causal mask (using KV caching).  

 7.2.2 Guided Sampling:  
     - Adjust logits using classifier‑free guidance (CFG):  
       * Scale logits based on current generation stage.  
     - Sample token indices via top‑k/top‑p methods.  

7.3 Convert sampled indices into token embeddings (via VQVAE codebook lookup).  


## 8. Multi‑Scale Progressive Generation 

8.1 Update Latent Representation (`f_hat`):  
  - Use VQVAE’s get_next_autoregressive_input to “add” the new token embeddings into f_hat.  

8.2 Upsampling:  
  - Interpolate f_hat to match the next scale’s resolution (defined by patch_nums).  

8.3 Prepare Next Token Map:  
  - Re-embed the updated f_hat (via word_embed) and add corresponding positional/level embeddings.  

## 9. Final Image Reconstruction via VQVAE  

9.1 After processing all scales, the final latent f_hat represents the full-resolution token map.  

9.2 Pass f_hat through post‑processing conv and the VQVAE decoder to reconstruct the high‑resolution image.  

9.3 De‑normalize output to [0, 1].  

## 10. Training vs. Inference Summary

• Training Mode:  
  - Teacher forcing is used to supply ground‑truth tokens, and cross‑entropy loss guides the transformer.  

• Inference Mode:  
  - The transformer generates tokens autoregressively using classifier‑free guidance and sampling, then feeds the tokens progressively into higher scales.  
