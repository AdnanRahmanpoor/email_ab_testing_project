import pandas as pd
import numpy as np
import random
import os
import time
import multiprocessing
from joblib import Parallel, delayed

def generate_user_batch(batch_size, start_id, include_nulls=True):
    """Generate a batch of user data with engagement metrics"""
    # Use thread-local random number generators for better parallel performance
    rng = np.random.RandomState(42 + start_id)
    random_state = random.Random(42 + start_id)
    
    # Create base data with thread-local RNG
    data = {
        'user_id': range(start_id, start_id + batch_size),
        'variant': rng.choice(['A', 'B', 'C'], size=batch_size, p=[0.33, 0.33, 0.34]),
        'send_time': rng.choice(['morning', 'afternoon', 'evening'], size=batch_size, p=[0.4, 0.2, 0.4]),
        'content_layout': rng.choice(['text-heavy', 'visual-heavy', 'balanced'], size=batch_size, p=[0.3, 0.4, 0.3]),
        'account_age': rng.randint(1, 365, size=batch_size),
        'feature_usage': rng.randint(0, 50, size=batch_size)
    }
    
    df = pd.DataFrame(data)
    
    # Assign user segments - vectorized approach instead of apply for speed
    df['user_segment'] = 'active'  # Default value
    df.loc[df['account_age'] < 30, 'user_segment'] = 'new'
    # For inactive users, create random mask for those with account_age > 180
    high_age_mask = df['account_age'] > 180
    inactive_mask = high_age_mask & (rng.random(size=len(df)) < 0.3)
    df.loc[inactive_mask, 'user_segment'] = 'inactive'
    
    # Create arrays for probabilities based on user segments and other factors
    open_probs = np.zeros(len(df))
    
    # Variant A
    mask_a = df['variant'] == 'A'
    mask_a_active = mask_a & (df['user_segment'] == 'active')
    mask_a_new = mask_a & (df['user_segment'] == 'new')
    mask_a_other = mask_a & ~mask_a_active & ~mask_a_new
    
    open_probs[mask_a_active] = 0.30
    open_probs[mask_a_new] = 0.20
    open_probs[mask_a_other] = 0.15
    
    # Variant B
    mask_b = df['variant'] == 'B'
    mask_b_new = mask_b & (df['user_segment'] == 'new')
    mask_b_active = mask_b & (df['user_segment'] == 'active')
    mask_b_other = mask_b & ~mask_b_new & ~mask_b_active
    
    open_probs[mask_b_new] = 0.28
    open_probs[mask_b_active] = 0.22
    open_probs[mask_b_other] = 0.12
    
    # Variant C (control)
    mask_c = df['variant'] == 'C'
    open_probs[mask_c] = 0.25
    
    # Adjust for send time
    send_time_factors = np.ones(len(df))
    send_time_factors[df['send_time'] == 'morning'] = 1.2
    send_time_factors[df['send_time'] == 'afternoon'] = 1.1
    send_time_factors[df['send_time'] == 'evening'] = 0.9
    
    # For variant B, evening performs better
    send_time_factors[mask_b & (df['send_time'] == 'evening')] = 1.05
    send_time_factors[mask_b & (df['send_time'] == 'morning')] = 0.95
    send_time_factors[mask_b & (df['send_time'] == 'afternoon')] = 1.1
    
    # Adjust for content layout
    layout_factors = np.ones(len(df))
    
    # Variant A performs better with visual-heavy
    layout_factors[mask_a & (df['content_layout'] == 'visual-heavy')] = 1.15
    layout_factors[mask_a & (df['content_layout'] == 'balanced')] = 1.0
    layout_factors[mask_a & (df['content_layout'] == 'text-heavy')] = 0.95
    
    # Variant B performs better with text-heavy
    layout_factors[mask_b & (df['content_layout'] == 'text-heavy')] = 1.1
    layout_factors[mask_b & (df['content_layout'] == 'balanced')] = 1.05
    layout_factors[mask_b & (df['content_layout'] == 'visual-heavy')] = 0.9
    
    # Calculate final open probabilities
    open_probs = open_probs * send_time_factors * layout_factors
    open_probs = np.minimum(open_probs, 0.99)  # Cap at 99%
    
    # Generate opens
    opens = rng.binomial(1, open_probs)
    
    # Generate clicks (only for opened emails)
    clicks = np.zeros(len(df))
    opened_indices = np.where(opens == 1)[0]
    
    # Base click probabilities
    click_probs = np.zeros(len(opened_indices))
    for i, idx in enumerate(opened_indices):
        if df.iloc[idx]['variant'] == 'A':
            click_probs[i] = 0.25 + (df.iloc[idx]['feature_usage'] / 200)
        elif df.iloc[idx]['variant'] == 'B':
            click_probs[i] = 0.20 + (df.iloc[idx]['feature_usage'] / 250)
        else:  # Control
            click_probs[i] = 0.15
    
    click_probs = np.minimum(click_probs, 0.9)  # Cap at 90%
    click_results = rng.binomial(1, click_probs)
    
    clicks[opened_indices] = click_results
    
    # Generate conversions (only for clicked emails)
    conversions = np.zeros(len(df))
    clicked_indices = np.where(clicks == 1)[0]
    
    # Base conversion probabilities
    convert_probs = np.zeros(len(clicked_indices))
    for i, idx in enumerate(clicked_indices):
        if df.iloc[idx]['variant'] == 'A':
            convert_probs[i] = 0.15 if df.iloc[idx]['user_segment'] == 'active' else 0.08
        elif df.iloc[idx]['variant'] == 'B':
            convert_probs[i] = 0.12 if df.iloc[idx]['user_segment'] == 'new' else 0.10
        else:  # Control
            convert_probs[i] = 0.05
    
    convert_results = rng.binomial(1, convert_probs)
    conversions[clicked_indices] = convert_results
    
    # Assign the results to the DataFrame
    df['open'] = opens
    df['click'] = clicks
    df['convert'] = conversions
    
    # Add timestamp column (emails sent over a 2-week period for large datasets)
    base_date = pd.Timestamp('2023-04-01')
    campaign_days = 14  # Use fixed campaign days for consistent timestamps across batches
    
    # Generate timestamps
    batch_timestamps = pd.date_range(
        start=base_date, 
        periods=batch_size, 
        freq=f'{(campaign_days*24*60*60)/batch_size:.0f}s'
    ).tolist()
    
    # Shuffle to avoid perfect chronological ordering
    random_state.shuffle(batch_timestamps)
    df['timestamp'] = batch_timestamps
    
    # Create temporary arrays for hour and minute adjustments
    hours = np.zeros(len(df), dtype=int)
    minutes = rng.randint(0, 60, size=len(df))
    
    # Adjust timestamps based on send_time
    morning_mask = df['send_time'] == 'morning'
    afternoon_mask = df['send_time'] == 'afternoon'
    evening_mask = df['send_time'] == 'evening'
    
    hours[morning_mask] = rng.randint(8, 12, size=morning_mask.sum())
    hours[afternoon_mask] = rng.randint(12, 17, size=afternoon_mask.sum())
    hours[evening_mask] = rng.randint(17, 22, size=evening_mask.sum())
    
    # Apply the hour and minute adjustments
    for i in range(len(df)):
        df.at[i, 'timestamp'] = df.at[i, 'timestamp'].replace(
            hour=int(hours[i]), 
            minute=int(minutes[i])
        )
    
    # Add device type
    df['device'] = rng.choice(
        ['mobile', 'desktop', 'tablet'], 
        size=len(df), 
        p=[0.55, 0.35, 0.10]
    )
    
    # Add some null values if requested - fast version
    if include_nulls:
        # Make ~2% of account_age null
        null_indices = rng.choice(df.index, size=int(len(df) * 0.02), replace=False)
        df.loc[null_indices, 'account_age'] = np.nan
        
        # Make ~1% of feature_usage null
        null_indices = rng.choice(df.index, size=int(len(df) * 0.01), replace=False)
        df.loc[null_indices, 'feature_usage'] = np.nan
        
        # Make ~0.5% of user_segment null
        null_indices = rng.choice(df.index, size=int(len(df) * 0.005), replace=False)
        df.loc[null_indices, 'user_segment'] = None
        
        # For ~1% of records that were opened, set to NaN to simulate tracking failures
        open_count = (df['open'] == 1).sum()
        if open_count > 0:
            open_indices = df.index[df['open'] == 1].tolist()
            sample_size = int(open_count * 0.01)
            if sample_size > 0:
                open_nulls = random_state.sample(open_indices, sample_size)
                df.loc[open_nulls, 'open'] = np.nan
    
    return df

def generate_email_data(n_users=500000, include_nulls=True, include_duplicates=True, output_file='email_campaign_data.csv'):
    """
    Generate a simulated email campaign dataset with:
    - Realistic open, click, and conversion rates
    - Some null values (if include_nulls=True)
    - Some duplicate records (if include_duplicates=True)
    - Optimized for large datasets (default 500,000 users)
    - Using aggressive parallelization for maximum speed
    """
    print(f"Generating data for {n_users} users with optimized parallel processing...")
    
    # For maximum speed, use more cores and smaller batches
    num_workers = min(multiprocessing.cpu_count(), 32)  # Use up to 32 workers
    batch_size = min(10000, max(1000, n_users // (num_workers * 2)))  # Smaller batches
    
    # Calculate how many complete batches we need
    complete_batches = n_users // batch_size
    remainder = n_users % batch_size
    
    # Set up batch parameters
    batch_params = []
    current_id = 1
    
    for i in range(complete_batches):
        batch_params.append((batch_size, current_id, include_nulls))
        current_id += batch_size
    
    # Add the remainder batch if needed
    if remainder > 0:
        batch_params.append((remainder, current_id, include_nulls))
    
    # Process batches in parallel with progress tracking
    start_time = time.time()
    print(f"Using {num_workers} worker threads with batch size of {batch_size}")
    
    # Use joblib for better parallel performance
    dfs = Parallel(n_jobs=num_workers, verbose=10)(
        delayed(generate_user_batch)(size, start_id, include_nulls) 
        for size, start_id, include_nulls in batch_params
    )
    
    print(f"Parallel processing complete. Combining {len(dfs)} batches...")
    # Combine all batches
    df = pd.concat(dfs, ignore_index=True)
    
    # Add duplicate records if requested - optimized version
    if include_duplicates:
        print("Adding duplicate records...")
        # For large datasets, limit duplicates to a reasonable number
        duplicate_count = min(int(len(df) * 0.03), 15000)
        # Use thread-local RNG
        thread_rng = np.random.RandomState(42)
        duplicate_indices = thread_rng.choice(df.index, size=duplicate_count, replace=False)
        duplicates = df.loc[duplicate_indices].copy()
        
        # Efficiently modify the duplicates to simulate data entry errors
        rand_mask = thread_rng.random(len(duplicates)) < 0.5
        duplicates.loc[rand_mask, 'send_time'] = thread_rng.choice(
            ['morning', 'afternoon', 'evening'], 
            size=rand_mask.sum()
        )
        
        rand_mask = thread_rng.random(len(duplicates)) < 0.3
        duplicates.loc[rand_mask, 'content_layout'] = thread_rng.choice(
            ['text-heavy', 'visual-heavy', 'balanced'], 
            size=rand_mask.sum()
        )
        
        # Combine original data with duplicates
        df = pd.concat([df, duplicates], ignore_index=True)
    
    # Ensure directory exists
    os.makedirs('data/raw', exist_ok=True)
    
    # For large datasets, use optimized CSV writing
    print(f"Saving {len(df)} records to CSV...")
    
    output_path = f'data/raw/{output_file}'

    chunk_size = 100000

    for i, start_idx in enumerate(range(0, len(df), chunk_size)):
        end_idx = min(start_idx + chunk_size, len(df))
        chunk = df.iloc[start_idx:end_idx]

        header = (i == 0)

        mode = 'w' if i == 0 else 'a'

        chunk.to_csv(output_path, mode=mode, header=header, index=False, escapechar='\\')
    
    print(f"Successfully generated and saved {len(df)} records")
    
    return df

if __name__ == "__main__":
    import time
    
    # Force NumPy to use multiple threads
    os.environ["OMP_NUM_THREADS"] = str(multiprocessing.cpu_count())
    os.environ["MKL_NUM_THREADS"] = str(multiprocessing.cpu_count())
    os.environ["OPENBLAS_NUM_THREADS"] = str(multiprocessing.cpu_count())
    
    start_time = time.time()
    print("Starting optimized parallel email campaign data generation...")
    
    # For testing with smaller dataset, uncomment below:
    # df = generate_email_data(n_users=10000)
    
    # For production-like dataset:
    df = generate_email_data(n_users=500000)
    
    duration = time.time() - start_time
    print(f"\nTotal execution time: {duration:.1f} seconds for {len(df)} records")
    print(f"Processing speed: {len(df)/duration:.1f} records per second")