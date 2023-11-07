def Nidealfac(ndim: int) -> float:
    """
    Choose factor of Nideal in g(r) calculation
    
    Inputs:
        ndim (int): system dimensionality

    Return:
        (float): Nideal
    """
    if ndim == 3:
        return 4.0 / 3
    elif ndim == 2:
        return 1.0 