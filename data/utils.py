from matplotlib import pyplot as plt


def log_3d(img, title="", file_name=None):
    center_slices = [dim // 2 for dim in img.shape]
    # img = np.transpose(img, (0, 2, 1))

    fig, axes = plt.subplots(1, 3, figsize=(6, 2))
    fig.suptitle(title, fontsize=16, fontweight='bold')
    # titles = ['Axial', 'Coronal', 'Sagittal']

    slices = [
        img[center_slices[0], :, :],
        img[:, center_slices[1], :],
        img[:, :, center_slices[2]]
    ]
    # cnt = 0
    for ax, slice_img in zip(axes, slices):
        # if cnt != 0:
        #     rotated_slice = slice_img
        # else:
        #     rotated_slice = np.rot90(slice_img, k=1)
        # rotated_slice = np.rot90(slice_img, k=1)
        ax.imshow(slice_img, cmap='gray')
        ax.set_facecolor('none')
        # ax.set_title(title)
        ax.axis('off')
        # cnt += 1

        # Save the figure to a file
    if file_name is None:
        plt.show()
    else:
        plt.savefig(f"{file_name}.png", transparent=True, bbox_inches='tight')
    plt.close(fig)
