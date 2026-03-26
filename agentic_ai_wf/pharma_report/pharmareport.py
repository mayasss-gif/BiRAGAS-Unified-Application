from report import pharmareport
disease_name = "Pancreatic Cancer"
path = {
    "Cohort":r"C:\Users\Raafeh\Downloads\Diabetesgeo.json",
    "Harmonization" : [r"C:\Users\Raafeh\Downloads\stats_filtered_Cohort_test_GSE40174_t0.30_n1845 (2).txt",r"C:\Users\Raafeh\Downloads\filtered_Cohort_test_GSE40174_t0.30_n1845.csv"],
    "DEG":[r"C:\Users\Raafeh\Downloads\DEGs.csv",disease_name],
    "Gene": r"C:\Users\Raafeh\Downloads\DEGs.csv",
    "Pathway":r"C:\Users\Raafeh\Downloads\pathway.csv",
    "Drug":r"C:\Users\Raafeh\Downloads\drug.csv"
}
output_dir = "Reports"
filepath = pharmareport(path,output_dir)
print(filepath)
