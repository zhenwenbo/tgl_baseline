import os
import re


def find_and_process_log_files(directory):
    """
    Recursively search for .log files in the given directory and its subdirectories.
    If the pattern is found in a file, merge all the lists into one and sort it in descending order,
    then append it to the end of the file.
    """
    # Pattern to match the text with multiple lists inside a single outer list
    pattern = r'total_test_res: \[\[(.*?)\]\]'
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.log'):
                file_path = os.path.join(root, file)
                with open(file_path, 'r') as f:
                    content = f.read()
                
                # Search for the pattern
                match = re.search(pattern, content, re.DOTALL)
                if match:
                    # Use regex to extract all numbers from the matched group
                    numbers_str = re.sub(r'[\'\"]|\[|\]', '', match.group(1))
                    numbers_str = re.split(r',\s*', numbers_str)
                    
                    # Convert strings to floats and combine into one list
                    combined_list = [float(num.strip()) for num in numbers_str if num.strip()]
                    
                    # Sort the combined list in descending order
                    sorted_combined_list = sorted(combined_list, reverse=True)
                    
                    # Prepare the result string
                    result_str = f"\nSorted combined results: {sorted_combined_list}\n"
                    
                    # Append the sorted results to the end of the file
                    with open(file_path, 'a') as f:
                        f.write(result_str)


if __name__ == "__main__":
    # Replace 'your_directory_here' with the path to the directory you want to search.
    find_and_process_log_files('/home/guorui/workspace/dgnn/exp/准确率结果')