import os
import pandas as pd

def get_age_class(age_in_months):
    """Map age in months to the appropriate age class."""
    if 0 <= age_in_months <= 36:
        return '0-3'
    elif 37 <= age_in_months <= 84:
        return '3-7'
    elif 85 <= age_in_months <= 120:
        return '7-10'
    elif 121 <= age_in_months <= 156:
        return '10-13'
    else:
        return 'Other'

def read_and_concat_data(folder_path, prefix, participant_number):
    """Read and concatenate data from a CSV file into a single comma-separated string."""
    file_path = os.path.join(folder_path, f'{prefix}{participant_number}.csv')
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        df = df.fillna('').astype(str)
        concatenated_string = ','.join(df.apply(lambda x: ','.join(x), axis=1))
        return concatenated_string
        #return ','.join(map(str, df.values.flatten()))
    else:
        return ''

def main():
    # Paths
    heart_folder = r'F:\radar_children\Children Dataset\FMCW Radar\Heart Rate & Breathing Rate'
    breath_folder = r'F:\radar_children\Children Dataset\FMCW Radar\Heart Rate & Breathing Rate'
    age_file_path = r'F:\radar_children\Children Dataset\Participant\Human Data\HumanData.xlsx'
    output_file_path = 'merged_1.csv'

    # Read age data
    age_df = pd.read_excel(age_file_path)
    
    # Initialize result list
    result_data = []

    for idx, row in age_df.iterrows():
        participant_number = row['Participant']  # Assuming the participant number is in the 'Participant' column
        age_in_months = row['Age (month)']  # Assuming the age is in the 'Age' column
        age_class = get_age_class(age_in_months)

        heart_data = read_and_concat_data(heart_folder, 'Heart_', participant_number)
        breath_data = read_and_concat_data(breath_folder, 'Breath_', participant_number)
        
        result_data.append([age_class, heart_data, breath_data])

    # Create DataFrame
    result_df = pd.DataFrame(result_data, columns=['Age_Class', 'Heart', 'Breath'])
    
    # Save to CSV
    result_df.to_csv(output_file_path, index=False)

if __name__ == "__main__":
    main()
