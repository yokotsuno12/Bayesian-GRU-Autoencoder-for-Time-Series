from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
import torch
import numpy as np
import sklearn.model_selection

class OurDataset(torch.utils.data.Dataset):
    def __init__(self, X_input, X_target):
        super(OurDataset, self).__init__()
        self.X_input = X_input
        self.X_target = X_target
        self.num_variables = (self.X_input).shape[2]

    def __len__(self):
        return (self.X_input).shape[0]

    def __getitem__(self, idx):
        return (self.X_input[idx,:,:], self.X_target[idx,:,:] )

def create_time_series_dataset(series, nav, nap, N):
    data = np.array(series)
    data = data.astype(np.float32)
    shape = data.shape
    if len(shape) == 1:
        length = len(data)
        data = data.reshape(length, 1)
        number_variables = 1
    elif len(shape) == 2:
        s1 = shape[1]
        s0 = shape[0]
        if s0<s1:
          data = data.transpose()
          length = s1
          number_variables = s0
        else:
          length = s0
          number_variables = s1
    else :
      print("problem with series size or shape")

    start_index = length - N*(nav + nap)

    if start_index < 0:
        raise ValueError("La série est trop courte pour les paramètres donnés.")

    X = np.zeros((N, nav, number_variables))
    y = np.zeros((N, nap, number_variables))

    for i in range(N):
        start_input = start_index + i*(nav + nap)
        end_input = start_input + nav
        X[i, :, :] = data[start_input:end_input][:]
        end_output = end_input + nap
        y[i, :, :] = data[end_input:end_output][:]

    return X, y

def prepare_univariate_data(series_data, nav, nap, train_size, test_size, batch_size, num_workers, create_val_loader=False, val_size=0, random_state=666, shuffle_split=True, seed_reproducibility=True):
    """
    Preprocesses univariate time series data for training, testing, and optional validation.

    Args:
        series_data (pd.Series or np.ndarray): The univariate time series.
        nav (int): Number of input time steps (N_input).
        nap (int): Number of output time steps (N_output).
        train_size (int): Number of samples for the training DataLoader.
        test_size (int): Number of samples for the testing DataLoader.
        batch_size (int): Batch size for the DataLoaders.
        num_workers (int): Number of worker processes for data loading.
        create_val_loader (bool): If True, a validation DataLoader is also created.
        val_size (int): Number of samples for the validation DataLoader (only if create_val_loader is True).
        random_state (int): Seed for random split for reproducibility.
        shuffle_split (bool): Whether to shuffle data before splitting (for train_test_split).

    Returns:
        tuple: A tuple containing (scaler, train_loader, test_loader) or (scaler, train_loader, test_loader, val_loader).
    """
    # 1. Normalize the data
    scaler = StandardScaler()
    if isinstance(series_data, np.ndarray):
        series_scaled = scaler.fit_transform(series_data.reshape(-1, 1))
    else:
        series_scaled = scaler.fit_transform(series_data.to_numpy().reshape(-1, 1))

    # 2. Create time series dataset
    N_total_samples = train_size + test_size
    if create_val_loader:
        if val_size == 0:
          raise ValueError("val_size must be greater than 0 if create_val_loader is True.")
        N_total_samples += val_size
    if N_total_samples > len(series_scaled):
        raise ValueError("The total number of samples exceeds the available data length.")

    X, y = create_time_series_dataset(series_scaled, nav, nap, N_total_samples)

    # 3. Train/Test Split
    X_train_full, X_test, y_train_full, y_test = sklearn.model_selection.train_test_split(
        X, y, test_size=test_size, train_size=train_size + (val_size if create_val_loader else 0),
        random_state=random_state, shuffle=shuffle_split, stratify=None
    )

    generator = torch.Generator()
    if seed_reproducibility:
        generator.manual_seed(0)
    else:
        generator.seed()

    if create_val_loader:
        # 4. Train/Validation Split
        X_train, X_val, y_train, y_val = sklearn.model_selection.train_test_split(
            X_train_full, y_train_full, test_size=val_size, train_size=train_size,
            random_state=random_state, shuffle=shuffle_split, stratify=None
        )

        our_dataset_train = OurDataset(X_train, y_train)
        our_dataset_val = OurDataset(X_val, y_val)
        train_loader = DataLoader(our_dataset_train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, generator=generator)
        val_loader = DataLoader(our_dataset_val, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, generator=generator)

    else:
        our_dataset_train = OurDataset(X_train_full, y_train_full)
        train_loader = DataLoader(our_dataset_train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, generator=generator)

    our_dataset_test = OurDataset(X_test, y_test)
    test_loader = DataLoader(our_dataset_test, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, generator=generator)

    if create_val_loader:
        return scaler, train_loader, test_loader, val_loader
    else:
        return scaler, train_loader, test_loader

def prepare_multivariate_data(series_list, nav, nap, train_size, test_size, batch_size, num_workers, create_val_loader=False, val_size=0, random_state=666, shuffle_split=True, seed_reproducibility=True):
    """
    Preprocesses multivariate time series data for training, testing, and optional validation.

    Args:
        series_list (list of pd.Series or np.ndarray): A list of univariate time series for multivariate input.
        nav (int): Number of input time steps (N_input).
        nap (int): Number of output time steps (N_output).
        train_size (int): Number of samples for the training DataLoader.
        test_size (int): Number of samples for the testing DataLoader.
        batch_size (int): Batch size for the DataLoaders.
        num_workers (int): Number of worker processes for data loading.
        create_val_loader (bool): If True, a validation DataLoader is also created.
        val_size (int): Number of samples for the validation DataLoader (only if create_val_loader is True).
        random_state (int): Seed for random split for reproducibility.
        shuffle_split (bool): Whether to shuffle data before splitting (for train_test_split).

    Returns:
        tuple: A tuple containing (train_loader, test_loader) or (train_loader, val_loader, test_loader).
    """
    scaled_series_list = []
    list_of_scalers = []
    if isinstance(series_list[0], np.ndarray):
        for series_data in series_list:
            scaler = StandardScaler()
            scaled_series_list.append(scaler.fit_transform(series_data.reshape(-1, 1)))
            list_of_scalers.append(scaler)
    else:
        for series_data in series_list:
            scaler = StandardScaler()
            scaled_series_list.append(scaler.fit_transform(series_data.to_numpy().reshape(-1, 1)))
            list_of_scalers.append(scaler)

    # Concatenate scaled series into a single multivariate array
    # Assuming all series have the same length
    multivariate_data = np.concatenate(scaled_series_list, axis=1)

    # 2. Create time series dataset
    N_total_samples = train_size + test_size
    if create_val_loader:
        N_total_samples += val_size
        if val_size == 0:
            raise ValueError("val_size must be greater than 0 if create_val_loader is True.")
    if N_total_samples > len(scaled_series_list[0]):
        raise ValueError("The total number of samples exceeds the available data length.")

    X, y = create_time_series_dataset(multivariate_data, nav, nap, N_total_samples)

    # 3. Train/Test Split
    X_train_full, X_test, y_train_full, y_test = sklearn.model_selection.train_test_split(
        X, y, test_size=test_size, train_size=train_size + val_size,
        random_state=random_state, shuffle=shuffle_split, stratify=None
    )

    train_loader = None
    val_loader = None

    generator = torch.Generator()
    if seed_reproducibility:
        generator.manual_seed(0)
    else:
        generator.seed()

    if create_val_loader:
        # 4. Train/Validation Split
        X_train, X_val, y_train, y_val = sklearn.model_selection.train_test_split(
            X_train_full, y_train_full, test_size=val_size, train_size=train_size,
            random_state=random_state, shuffle=shuffle_split, stratify=None
        )

        our_dataset_train = OurDataset(X_train, y_train)
        our_dataset_val = OurDataset(X_val, y_val)
        train_loader = DataLoader(our_dataset_train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, generator = generator)
        val_loader = DataLoader(our_dataset_val, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, generator = generator)

    else:
        our_dataset_train = OurDataset(X_train_full, y_train_full)
        train_loader = DataLoader(our_dataset_train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, generator = generator)

    our_dataset_test = OurDataset(X_test, y_test)
    test_loader = DataLoader(our_dataset_test, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, generator=generator)

    if create_val_loader:
        return list_of_scalers, train_loader, val_loader, test_loader
    else:
        return list_of_scalers, train_loader, test_loader