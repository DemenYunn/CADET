def main():
    """Read and print the contents of a file."""
    try:
        # Get file path from user
        file_path = input("Enter the path to the file: ").strip('"')
        
        # Read and print file contents
        with open(file_path, 'r') as file:
            contents = file.read()
            print("\nFile contents:")
            print("-" * 40)
            print(contents)
            print("-" * 40)
            
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
