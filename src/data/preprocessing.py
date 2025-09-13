import torch


def flatten_latents(latents, labels):
    flat_set = []
    for latent in latents:
        flat = torch.cat([torch.flatten(latent['model_y']), torch.flatten(latent['model_uv'])]).cpu().numpy()
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

def create_patches_dataset(latents, labels, patch_size=1):
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

    for latent, label in zip(latents, labels):

        latent_y = latent['model_y'][0] # to remove batch size
        latent_uv = latent['model_uv'][0]
        
        patch_y  = get_single_patch(latent_y, 8,8)  
        patch_uv = get_single_patch(latent_uv, 8,8)


        patch = torch.cat((torch.flatten(patch_y), torch.flatten(patch_uv))).numpy()
        X_patches.append(patch)
        y_patches.append(label)
    
    print(f"Created {len(X_patches)} patches from {len(latents)} latents.")
    print(f"Each patch has shape: {X_patches[0].shape}")
    print(f"Labels shape: {len(y_patches)}")
    
    return X_patches, y_patches

#def create_test_patches_dataset(latents, labels, patch_size=1):
    """
    Creates a dataset of patches from the latents
    Args:
        latents: list of latent tensors of shape (1,C, H, W)
        labels: list of labels
        patch_size: size of the patch (default 1x1)
    Returns:
        X_grouped: list of lists, where each sublist contains patches for one image
        y_grouped: list of labels for each image
    """
    X_grouped = []
    y_grouped = []

    for latent, label in zip(latents, labels):

        latent_y = latent['model_y'][0] # to remove batch size
        latent_uv = latent['model_uv'][0]
        
        patch_y  = get_single_patch(latent_y, 8,8)  
        patch_uv = get_single_patch(latent_uv, 8,8)


        patch = torch.cat((torch.flatten(patch_y), torch.flatten(patch_uv))).numpy()
        
        X_grouped.append([patch])  # Wrap patch in a list to maintain structure
        y_grouped.append(label)
    
    print(f"Created {len(X_grouped)} grouped patches from {len(latents)} latents.")
    print(f"Each patch has shape: {X_grouped[0][0].shape}")
    print(f"Labels shape: {len(y_grouped)}")
    
    return X_grouped, y_grouped

def channels_to_tensor(latents,labels, channels_per_image=5):# TODO: REMOVE

    # TODO: Aggiungi descrizione
    if len(latents) != len(labels):
        raise ValueError("Latents and labels must have same size")

    X, new_labels = [],[]

    k = channels_per_image

    for latent,label in zip(latents,labels):
        #print(latent['model_y'].shape)
        latent_y = latent['model_y'][0] # to remove batch size
        latent_uv = latent['model_uv'][0] # TODO: prendere anche UV concatenando?

        latent = latent_y
        #print(latent_y.shape)
        #print(latent_uv.shape)

        #print(latent.shape)
        chls_idxs = torch.randperm(latent.shape[0])[:k]
        #print(chls_idxs)

        for ch in chls_idxs:
            ch_grid = latent[ch]
            flat = torch.flatten(ch_grid).numpy()
            X.append(flat)
            new_labels.append(label)
            #print(label, end=" ")
        #print("\n")
    return X, new_labels

#def channels_to_tensor_for_testing(latents, labels, channels_per_image=14):
    """
    Versione per testing che mantiene la struttura per immagine
    per permettere il voto a maggioranza
    
    Returns:
        X_grouped: lista di liste, dove ogni sottolista contiene i canali di un'immagine
        labels: labels originali (una per immagine)
    """
    if len(latents) != len(labels):
        raise ValueError("Latents and labels must have same size")

    X_grouped = []
    k = channels_per_image

    for latent, label in zip(latents, labels):
        latent_y = latent['model_y'][0]  # rimuovi batch size
        latent_uv = latent['model_uv'][0]  # TODO: considera anche UV se necessario
        
        latent = latent_y
        
        # Estrai k canali casuali
        chls_idxs = torch.randperm(latent.shape[0])[:k]
        
        # Lista per i canali di questa immagine
        image_channels = []
        for ch in chls_idxs:
            ch_grid = latent[ch]
            flat = torch.flatten(ch_grid).numpy()
            image_channels.append(flat)
        
        X_grouped.append(image_channels)
    
    return X_grouped, labels
