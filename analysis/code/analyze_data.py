import pandas as pd
import numpy as np
from linearmodels import PanelOLS

### DEFINE
def main():
    df = import_data()
    fit = run_regression(df)
    df_1960 = df[df['year']>=1960]
    fit1960 = run_regression(df_1960)
    formatted = format_model(fit)
    formatted_1960 = format_model(fit1960)
    fit_cluster = run_regression_cluster(df)
    formatted_cluster = format_model(fit_cluster)
    fit_1960_cluster = run_regression_cluster(df_1960)
    formatted_cluster_1960 = format_model(fit_1960_cluster)
    
    with open('output/regression.csv', 'w') as f:
        f.write('<tab:regression>' + '\n')
        formatted.to_csv(f, sep = '\t', index = False, header = False)
     
    with open('output/regression_1960.csv', 'w') as f:
        f.write('<tab:regression_1960>' + '\n')
        formatted_1960.to_csv(f, sep = '\t', index = False, header = False)
        
    with open('output/regression_cluster.csv', 'w') as f:
        f.write('<tab:regression_cluster>' + '\n')
        formatted_cluster.to_csv(f, sep = '\t', index = False, header = False)
        
    with open('output/regression_cluster_1960.csv', 'w') as f:
        f.write('<tab:regression_cluster_1960>' + '\n')
        formatted_cluster_1960.to_csv(f, sep = '\t', index = False, header = False)

    
def import_data():
    df = pd.read_csv('input/data_cleaned.csv')
    df['post_tv'] = df['year'] > df['year_tv_introduced']
    
    return(df)

def run_regression(df):
    df = df.set_index(['county_id', 'year'])
    model = PanelOLS.from_formula('chips_sold ~ 1 + post_tv + EntityEffects + TimeEffects', data = df)
    fit = model.fit()
    
    return(fit)
    
def run_regression_cluster(df):
    df = df.set_index(['county_id', 'year'])
    model = PanelOLS.from_formula('chips_sold ~ 1 + post_tv + EntityEffects + TimeEffects', data = df)
    fit = model.fit(cov_type='clustered', cluster_entity = True)
    
    return(fit)
     
    
def format_model(fit):
    formatted = pd.DataFrame({'coef'     : fit.params, 
                              'std_error': fit.std_errors, 
                              'p_value'  : fit.pvalues})
    formatted = formatted.loc[['post_tv']]
    
    return(formatted)  
       
       
    
### EXECUTE
main()