import pandas as pd

file_path = "/home/tongye/code_generation/pie-perf/data/improvement_pairs_additional_metadata_unpivoted_5_10_23.csv"

def save_chunk_to_csv(chunk, chunk_index):
    chunk_file_path = f'chunk_{chunk_index}.csv'
    chunk.to_csv(chunk_file_path, index=False)
    print(f'Saved chunk {chunk_index} to {chunk_file_path}')

def process_large_csv(file_path):
    chunk_size = 100  # 指定块的大小
    for chunk in pd.read_csv(file_path, chunksize=chunk_size):
        chunk_index = 1
        # 处理每个块的数据
        save_chunk_to_csv(chunk, chunk_index)
        assert False


if __name__ == "__main__":
    process_large_csv(file_path)