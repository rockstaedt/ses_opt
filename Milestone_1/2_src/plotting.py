import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

###############################################################################
### Matplotlib Settings
###############################################################################

settings = {
    'text.usetex': True,
    'font.weight' : 'normal',
    'font.size'   : 20
}
plt.rcParams.update(**settings)

# resolution for plots
dpi = 300

# path to csv files
path = '../3_results/results_'

# saving path
saving_path = '../4_plots/'

###############################################################################
### Helper
###############################################################################

def create_generator_df(lamda:int = 0.2197, unit:int = 0) -> pd.DataFrame:
    df = pd.read_csv(
        path + 'generator_lambda_' + str(lamda) + '.csv'
    )
    df = df[df.Generator == unit]
    return df

def create_retailer_df(lamda:int = 0.2197) -> pd.DataFrame:
    return pd.read_csv(
        path + 'retailer_lambda_' + str(lamda) + '.csv'
    )

def get_demand() -> np.array:
    return np.array(
        [8,8,10,10,10,16,22,24,26,32,30,28,22,18,16,16,20,24,28,34,38,30,22,12]
    )

###############################################################################
### Solution for lambda 21.97 ct/kWh
###############################################################################

lamda = 0.2197

# get generator specific dataframes
df_generator_1 = create_generator_df(lamda=lamda, unit=0)
df_generator_2 = create_generator_df(lamda=lamda, unit=1)

# retailer
df_retailer = create_retailer_df(lamda=lamda)

# hourly time steps
t = np.array(df_generator_1.Hour.values) + 1

# demand
demand = get_demand()

# import
df_retailer['Imports'] = df_retailer.apply(
    lambda x: x['Import/Export'] if x['Import/Export'] > 0 else 0,
    axis=1
)
import_values = df_retailer['Imports'].values

# export
df_retailer['Exports'] = df_retailer.apply(
    lambda x: x['Import/Export'] if x['Import/Export'] < 0 else 0,
    axis=1
)
export_values = df_retailer['Exports'].values

# production
production_stacked = np.vstack(
    [
        list(df_generator_1.Generation),
        list(df_generator_2.Generation),
        import_values
    ]
)

# consumption
consumption_stacked = np.vstack(
    [
        (
            np.array(list(df_generator_1.Generation))
            + np.array(list(df_generator_2.Generation))
            + np.array(export_values)
            + np.array(import_values)
        )*(-1),
        export_values
    ]
)

# plot
fig, ax = plt.subplots(figsize=(12,8))
# production
ax.stackplot(
    t,
    production_stacked,
    labels=['Generator 1', 'Generator 2', 'Import'],
    colors=['mediumblue', 'forestgreen', 'darkorange']
)
# consumption
ax.stackplot(
    t,
    consumption_stacked,
    colors=['grey', 'lightseagreen'],
    labels=['', 'Export']
)
# demnad
ax.plot(t, demand*(-1), linestyle='--', color='red', label='Demand')
plt.hlines(0, xmin=1, xmax=len(t),linestyles='-', linewidth=2.0)
ax.set_xlabel('Hourly timesteps')
ax.set_ylabel('kW')
#ax.set_title(model_type_to_description[model_type] + ' - ' + time_period)
ax.grid()

plt.legend(bbox_to_anchor=(1,1), loc="upper left")
plt.savefig(
    saving_path + f'balance_plot_lambda_{str(lamda)}.png',
    dpi=dpi,
    bbox_inches='tight'
)

###############################################################################
### Solution for sensitivity analysis
###############################################################################

lamdas = np.arange(5,80,10)*0.01

fig, ax = plt.subplots(4,2,figsize=(26,16))

row = 0
col = 0

for lamda in lamdas:

    # get generator specific dataframes
    df_generator_1 = create_generator_df(lamda=round(lamda, 4), unit=0)
    df_generator_2 = create_generator_df(lamda=round(lamda, 4), unit=1)

    # retailer
    df_retailer = create_retailer_df(lamda=round(lamda, 4))

    # hourly time steps
    t = np.array(df_generator_1.Hour.values) + 1

    # demand
    demand = get_demand()

    # import
    df_retailer['Imports'] = df_retailer.apply(
        lambda x: x['Import/Export'] if x['Import/Export'] > 0 else 0,
        axis=1
    )
    import_values = df_retailer['Imports'].values

    # export
    df_retailer['Exports'] = df_retailer.apply(
        lambda x: x['Import/Export'] if x['Import/Export'] < 0 else 0,
        axis=1
    )
    export_values = df_retailer['Exports'].values

    # production
    production_stacked = np.vstack(
        [
            list(df_generator_1.Generation),
            list(df_generator_2.Generation),
            import_values
        ]
    )

    # consumption
    consumption_stacked = np.vstack(
        [
            (
                np.array(list(df_generator_1.Generation))
                + np.array(list(df_generator_2.Generation))
                + np.array(export_values)
                + np.array(import_values)
            )*(-1),
            export_values
        ]
    )

    # plots

    # production
    ax[row][col].stackplot(
        t,
        production_stacked,
        labels=['Generator 1', 'Generator 2', 'Import'],
        colors=['mediumblue', 'forestgreen', 'darkorange']
    )
    # consumption
    ax[row][col].stackplot(
        t,
        consumption_stacked,
        colors=['grey', 'lightseagreen'],
        labels=['', 'Export']
    )
    # demnad
    ax[row][col].plot(
        t,
        demand*(-1),
        linestyle='--',
        color='red',
        label='Demand'
    )
    ax[row][col].set_title(f'Lambda: {round(lamda, 4)}')
    #ax.set_title(model_type_to_description[model_type] + ' - ' + time_period)
    ax[row][col].grid()

    col += 1
    if col > 1:
        row += 1
        col = 0

# labels for axis
ax[3][0].set_xlabel('Hourly timesteps')
ax[3][1].set_xlabel('Hourly timesteps')
ax[0][0].set_ylabel('kW')
ax[1][0].set_ylabel('kW')
ax[2][0].set_ylabel('kW')
ax[3][0].set_ylabel('kW')

# legend for one plot
ax[0][1].legend(
    loc = "upper left",
    bbox_to_anchor=(1,1)
)

# for better layout
fig.tight_layout()

plt.savefig(
    saving_path + 'balance_plot_sensitivity.png',
    dpi=dpi,
    bbox_inches='tight'
)