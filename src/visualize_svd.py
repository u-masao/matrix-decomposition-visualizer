import matplotlib.pyplot as plt
import numpy as np

def visualize_svd(U, S, V):
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    
    # Visualize U
    axs[0].imshow(np.real(U), aspect='equal', cmap='gray')
    axs[0].set_title('U (左特異ベクトル)')
    
    # Visualize S
    axs[1].imshow(np.diag(S), aspect='equal', cmap='gray')
    axs[1].set_title('S (特異値)')
    
    # Visualize V
    axs[2].imshow(np.real(V), aspect='equal', cmap='gray')
    axs[2].set_title('V (右特異ベクトル)')
    
    plt.show()
