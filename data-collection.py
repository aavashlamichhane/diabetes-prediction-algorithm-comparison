from ucimlrepo import fetch_ucirepo

# fetch dataset
diabetes_130_us_hospitals_for_years_1999_2008 = fetch_ucirepo(id=296)
diabetes_130_us_hospitals_for_years_1999_2008.data.original.to_csv("diabetes_data.csv", index=False)