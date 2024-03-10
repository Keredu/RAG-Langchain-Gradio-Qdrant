import ftplib
import os
import time

def is_file(line):
    # This function checks if the line represents a file
    return line.startswith('-')

def download_PMC_pdfs_from_FTP(local_directory='data/PMC/'):
    """
    Download PDF files from a specified directory on an FTP server.
    """
    ftp = ftplib.FTP("ftp.ncbi.nlm.nih.gov")
    ftp.login()
    ftp.cwd('/pub/pmc/oa_pdf/')

    # Ensure the local directory exists
    os.makedirs(local_directory, exist_ok=True)

    directories = ftp.nlst()
    files_downloaded = 0
    max_files = 20
    
    for directory in directories:
        if files_downloaded >= max_files:
            break  # Exit if we've downloaded the maximum number of files

        try:
            # Skip if it's not a directory
            if '.' in directory:
                continue

            ftp.cwd(directory)
            file_lines = []
            ftp.dir(lambda line: file_lines.append(line))
            
            for line in file_lines:
                if files_downloaded >= max_files:
                    break  # Stop if max count is reached within the inner loop
                
                if is_file(line):
                    filename = line.split()[-1]
                    local_filename = os.path.join(local_directory, filename)
                    with open(local_filename, 'wb') as file:
                        ftp.retrbinary('RETR ' + filename, file.write)
                    print(f"Downloaded: {filename}")
                    files_downloaded += 1
                    time.sleep(1)  # Pause between downloads

            ftp.cwd('..')  # Go back to the parent directory
        except ftplib.error_perm as e:
            print(f"FTP Error: {e}")
        except Exception as e:
            print(f"General Error: {e}")
        finally:
            # This ensures the FTP connection is reset if an error occurs
            if ftp.sock:
                ftp.close()
                ftp.connect("ftp.ncbi.nlm.nih.gov")
                ftp.login()
                ftp.cwd('/pub/pmc/oa_pdf/' + directory)
    
    ftp.quit()
    print(f"Finished downloading {files_downloaded} files.")

if __name__ == '__main__':
    download_PMC_pdfs_from_FTP()
