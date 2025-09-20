import torch


def flatten_latents(latents, labels, concat_uv=True):
    flat_set = []
    for latent in latents:
        if concat_uv:
            flat = torch.cat([torch.flatten(latent['model_y']), torch.flatten(latent['model_uv'])]).cpu().numpy()
        else:
            flat = torch.flatten(latent['model_y']).cpu().numpy()
        flat_set.append(flat)
    return flat_set, labels

def get_single_patch(latent, row, col):
    """
    Given a latent tensor of shape (C, H ,W) returns the concatenations of patches in (x,y) position
    of shape (C, 1, 1)
    Args:
        latent: tensor of shape (C, H, W)
    Returns:
        patch: tensor of shape (C, 1, 1) 
    """
    C, H, W = latent.shape
    
    if row >= H or col >= W:
        raise ValueError("row and col must be less than H and W respectively")
    
    patch = latent[:, row, col]  
    return patch


def get_center_nn_points(H,W):
    """
    Return the coordinates of the center and its nearest neighbors in a square patch
    """
    cx = H // 2
    cy = W // 2
    x = cx - 1
    y = cy - 1
    return [(i, j) for i in range(x, x+3) for j in range(y, y+3)]


def create_patches_dataset(latents, labels, concat_uv=True, patch_num=1, not_all_patches=True):
    """
    Creates a dataset of patches from the latents
    Args:
        latents: list of latent tensors of shape (1,C, H, W)
        labels: list of labels
        patch_size: size of the patch (default 1x1)
    Returns:
        X_patches: list of patches of shape (C*patch_size*patch_size,)
        y_patches: list of labels for each patch
    """
    X_patches = []
    y_patches = []

    _, C, H, W = latents[0]['model_y'].shape  # assuming all latents have the same shape 

    
    x_center = H // 2
    y_center = W // 2
    if patch_num==1:
        points = [(x_center, y_center)]
    else:
        import random
        if not_all_patches:
            random.seed(42) # for reproducibility and testing
        points = [(random.randint(0, H-1), random.randint(0, W-1)) for _ in range(patch_num)] # same points for all latents
        #points = get_center_nn_points(H,W)
    
    for latent, label in zip(latents, labels):

        latent_y = latent['model_y'][0] # to remove batch size
        latent_uv = latent['model_uv'][0]
        
        for x, y in points:
    
            patch_y  = get_single_patch(latent_y, x, y)
            if concat_uv:
                patch_uv = get_single_patch(latent_uv, x, y)
                patch = torch.cat((torch.flatten(patch_y), torch.flatten(patch_uv))).numpy()
            else:
                patch = torch.flatten(patch_y).numpy()

            X_patches.append(patch)
            y_patches.append(label)

    return X_patches, y_patches


def YUV_single_patch_per_latent(latents, labels):
    return create_patches_dataset(latents, labels, patch_num=1)
def Y_single_patch_per_latent(latents, labels):
    return create_patches_dataset(latents, labels, concat_uv=False, patch_num=1)



patch_num_default = 15
def Y_multiple_patches_per_latent(latents, labels, patch_num=patch_num_default):
    return create_patches_dataset(latents, labels, concat_uv=False, patch_num=patch_num)
def YUV_multiple_patches_per_latent(latents, labels, patch_num=patch_num_default):
    return create_patches_dataset(latents, labels, patch_num=patch_num)




def Y_all_patches_per_latent(latents, labels):
    C, H, W = latents[0]['model_y'].shape[1:]  # assuming all latents have the same shape
    return create_patches_dataset(latents, labels, concat_uv=False, patch_num=H*W, not_all_patches=False)
def YUV_all_patches_per_latent(latents, labels):
    C, H, W = latents[0]['model_y'].shape[1:]  # assuming all latents have the same shape
    return create_patches_dataset(latents, labels, patch_num=H*W, not_all_patches=False)