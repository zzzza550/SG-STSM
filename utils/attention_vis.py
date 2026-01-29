import matplotlib.pyplot as plt

def visualize_attention(attn_s_map, attn_t_map, save_path=None):
    # Spatial
    plt.figure(figsize=(6, 5))
    plt.imshow(attn_s_map, cmap='hot')
    plt.colorbar()
    plt.title("Spatial Attention (Joint-to-Joint)")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path + "_spatial.png")
    plt.close()

    # Temporal
    plt.figure(figsize=(6, 5))
    plt.imshow(attn_t_map, cmap='hot')
    plt.colorbar()
    plt.title("Temporal Attention (Frame-to-Frame)")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path + "_temporal.png")
    plt.close()
