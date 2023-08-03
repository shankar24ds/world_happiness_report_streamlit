import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from IPython.display import display     
import os
import warnings
warnings.filterwarnings("ignore")

def explore_data():
    """
    Explore the World Happiness DataFrame with various visualizations.

    This function reads the DataFrame from the specified input CSV file,
    performs exploratory data analysis, and displays various visualizations.

    Parameters:
        None

    Returns:
        None
    """

    # importing the file needed for the analysis
    df = pd.read_pickle("data/processed/df_cleaned.pkl")

    ## 1. Table - Summary Statistics
    # yearby stats
    result_wide = df.groupby('year').agg({'score' : ['max', 'min', np.mean, np.std, np.median],
                                        'gdp_per_capita' : ['max', 'min', np.mean, np.std, np.median],
                                        'social_support' : ['max', 'min', np.mean, np.std, np.median],
                                        'life_expectancy' : ['max', 'min', np.mean, np.std, np.median],
                                        'freedom' : ['max', 'min', np.mean, np.std, np.median],
                                        'generosity': ['max', 'min', np.mean, np.std, np.median],
                                        'corruption': ['max', 'min', np.mean, np.std, np.median]
                                        })
    display(result_wide.T)

    ## 2.1 Histogram - Score - Yearwise
    fig = make_subplots(rows=3, cols=2, subplot_titles=("2015", "2016", "2017", "2018", "2019"))

    fig.add_trace(
        go.Histogram(x=df[df['year'] == 2015]['score'], marker_color="skyblue"),
        row=1, col=1
    )
    fig.add_trace(
        go.Histogram(x=df[df['year'] == 2016]['score'], marker_color="olive"),
        row=1, col=2
    )
    fig.add_trace(
        go.Histogram(x=df[df['year'] == 2017]['score'], marker_color="gold"),
        row=2, col=1
    )
    fig.add_trace(
        go.Histogram(x=df[df['year'] == 2018]['score'], marker_color="teal"),
        row=2, col=2
    )
    fig.add_trace(
        go.Histogram(x=df[df['year'] == 2019]['score'], marker_color="orange"),
        row=3, col=1
    )

    fig.update_layout(
        height=800,
        width=800,
        title="Histograms of Happiness Score by Year",
        xaxis=dict(title="Happiness Score"),
        yaxis=dict(title="Country Frequency"),
        showlegend=False,
        template="plotly_white"
    )

    fig.show()

    ## 2.2 Histogram(with 3 bins) - Score - Yearwise
    fig = make_subplots(rows=3, cols=2, subplot_titles=("2015", "2016", "2017", "2018", "2019"))

    num_bins = 3

    fig.add_trace(
        go.Histogram(x=df[df['year'] == 2015]['score'], marker_color="skyblue", nbinsx=num_bins),
        row=1, col=1
    )
    fig.add_trace(
        go.Histogram(x=df[df['year'] == 2016]['score'], marker_color="olive", nbinsx=num_bins),
        row=1, col=2
    )
    fig.add_trace(
        go.Histogram(x=df[df['year'] == 2017]['score'], marker_color="gold", nbinsx=num_bins),
        row=2, col=1
    )
    fig.add_trace(
        go.Histogram(x=df[df['year'] == 2018]['score'], marker_color="teal", nbinsx=num_bins),
        row=2, col=2
    )
    fig.add_trace(
        go.Histogram(x=df[df['year'] == 2019]['score'], marker_color="orange", nbinsx=num_bins),
        row=3, col=1
    )

    fig.update_layout(
        height=800,
        width=800,
        title="Histograms of Happiness Score by Year",
        xaxis=dict(title="Happiness Score"),
        yaxis=dict(title="Country Frequency"),
        showlegend=False,
        template="plotly_white"
    )

    fig.show()

    ## 2.3 KDE Plot - Score - Yearwise
    fig, ax = plt.subplots(figsize=(10, 6))

    sns.kdeplot(data=df[df['year'] == 2015], x="score", color="skyblue", label="2015")
    sns.kdeplot(data=df[df['year'] == 2016], x="score", color="olive", label="2016")
    sns.kdeplot(data=df[df['year'] == 2017], x="score", color="gold", label="2017")
    sns.kdeplot(data=df[df['year'] == 2018], x="score", color="teal", label="2018")
    sns.kdeplot(data=df[df['year'] == 2019], x="score", color="orange", label="2019")

    years = [2015, 2016, 2017, 2018, 2019]
    colors = ["skyblue", "olive", "gold", "teal", "orange"]
    for count, year in enumerate(years):
        mean_score = df[df['year'] == year]['score'].mean()
        ax.axvline(x=mean_score, color=colors[count], linestyle='dashed', label=f"Mean {year}: {mean_score:.2f}")

    ax.set_xlabel("Happiness Score")
    ax.set_ylabel("Density")
    ax.set_title("KDE Plots of Happiness Score by Year with Mean Lines")
    ax.legend()
    plt.show()

    ## 3. Box Plot - Score - Yearwise
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='year', y='score', data=df)
    plt.title('Box Plot of Score by Year')
    plt.xlabel('Year')
    plt.ylabel('Score')
    plt.show()

    ## 4. Line Plot
    def lineplot(factor):
        summary_df = df.groupby('year')[factor].agg(['max', 'min', 'mean']).reset_index()

        summary_df_melted = summary_df.melt(id_vars='year', var_name='Metrics', value_name=factor)

        fig = px.line(summary_df_melted, x='year', y=factor, color='Metrics', markers=True,
                    labels={'year': 'Year', factor: factor})

        return fig

    # List of factors to plot
    factors_to_plot = ['score', 'gdp_per_capita', 'social_support', 'life_expectancy', 'freedom', 'generosity', 'corruption']

    # Define relative heights for the rows (adjust as needed)
    heights = [0.3, 0.3, 0.2, 0.2]

    # Create subplots with appropriate-sized graphs
    fig = make_subplots(rows=4, cols=2, shared_yaxes=False,
                        subplot_titles=[f'{factor.capitalize()}' for factor in factors_to_plot],
                        horizontal_spacing=0.15, vertical_spacing=0.2,
                        row_heights=heights)

    for i, factor in enumerate(factors_to_plot, 1):
        line_plot = lineplot(factor)
        for trace in line_plot.data:
            fig.add_trace(go.Scatter(trace), row=(i-1)//2 + 1, col=(i-1)%2 + 1)
        fig.update_xaxes(title_text="Year", row=(i-1)//2 + 1, col=(i-1)%2 + 1)
        fig.update_yaxes(title_text=factor, row=(i-1)//2 + 1, col=(i-1)%2 + 1)

    # Update subplot settings for better appearance
    fig.update_layout(title_text="Line Plots of Factors Year-wise", showlegend=False,
                    width=1000, height=1200, font=dict(size=12))

    fig.show()

    ## 5.1 Table - Top 10 Countries
    top_rank = df[df['rank']<=10].sort_values(['year', 'rank'])
    top_rank = top_rank[['year', 'rank', 'country']]
    pivoted_df_top = top_rank.pivot(index='rank', columns='year', values='country').sort_index()
    display(pivoted_df_top)

    ## 5.2 Table - Bottom 10 Countries
    df_sorted = df[['year', 'rank', 'country']].sort_values(['year', 'rank'])
    subset_tail_df = df_sorted.groupby('year').tail(10)
    subset_tail_df = subset_tail_df[['year', 'country']].reset_index(drop=True)
    subset_tail_df['rank'] = subset_tail_df.groupby(['year']).cumcount() + 1
    pivoted_df_bottom = subset_tail_df.pivot(index='rank', columns='year', values='country').sort_index()
    display(pivoted_df_bottom)

    ## 6.1 Heatmap - Overall
    heatmap_data = df[['score', 'gdp_per_capita', 'social_support', 'life_expectancy', 'freedom', 'generosity', 'corruption']]
    df_corr = heatmap_data.corr().round(1)  
    mask = np.zeros_like(df_corr, dtype=bool)
    mask[np.triu_indices_from(mask)] = True
    df_corr_viz = df_corr.mask(mask).dropna(how='all').dropna('columns', how='all')
    fig = px.imshow(df_corr_viz, text_auto=True)

    fig.update_layout(
        height=800,
        width=800,
        title="Correlation between factors",
        showlegend=True,
        template="plotly_white"
    )

    fig.show()

    ## 6.2 Heatmap - YearWise
    fig = make_subplots(rows=3, cols=2, subplot_titles=[str(year) for year in df['year'].unique()])

    for i, year in enumerate(df['year'].unique()):
        heatmap_data = df[df['year'] == year][['score', 'gdp_per_capita', 'social_support', 'life_expectancy', 'freedom', 'generosity', 'corruption']]
        df_corr = heatmap_data.corr().round(1)
        mask = np.zeros_like(df_corr, dtype=bool)
        mask[np.triu_indices_from(mask)] = True
        df_corr_viz = df_corr.mask(mask).dropna(how='all').dropna('columns', how='all')

        fig.add_trace(
            go.Heatmap(z=df_corr_viz.values,
                    x=df_corr_viz.columns,
                    y=df_corr_viz.index,
                    zmin=-1,
                    zmax=1,
                    colorscale='RdBu',
                    colorbar=dict(len=0.2, y=0.85)),
            row=(i // 2) + 1,
            col=(i % 2) + 1
        )

    fig.update_layout(
        height=1000,
        width=1000,
        title="Correlation between factors for each year",
        showlegend=False,
        template="plotly_white"
    )

    fig.show()

    ## 7. Parallel Coordinates Plot - Year 2019
    df_2019 = df[df['year']==2019]
    # Let's reverse the min and max values for the Rank, so that the country with top rank comes on the top. 
    dimensions = list([ dict(range=(df_2019['rank'].max(), df_2019['rank'].min()), tickvals = df_2019['rank'], ticktext = df_2019['country'],label='country', values=df_2019['rank']),
                        dict(range=(df_2019['score'].min(),df_2019['score'].max()),label='score', values=df_2019['score']),
                        dict(range=(df_2019['gdp_per_capita'].min(),df_2019['gdp_per_capita'].max()),label='gdp_per_capita', values=df_2019['gdp_per_capita']),
                        dict(range=(df_2019['social_support'].min(), df_2019['social_support'].max()),label='social_support', values=df_2019['social_support']),
                        dict(range=(df_2019['life_expectancy'].min(), df_2019['life_expectancy'].max()), label='life_expectancy', values=df_2019['life_expectancy']),
                        dict(range=(df_2019['freedom'].min(), df_2019['freedom'].max()),label='freedom', values=df_2019['freedom']),
                        dict(range=(df_2019['generosity'].min(), df_2019['generosity'].max()),label='generosity', values=df_2019['generosity']),
                        dict(range=(df_2019['corruption'].min(), df_2019['corruption'].max()),label='corruption', values=df_2019['corruption']),
                    ])
    fig = go.Figure(data= go.Parcoords(line = dict(color = df_2019['rank'], colorscale = 'agsunset'), dimensions = dimensions))
    fig.update_layout(width=1200, height=1300,margin=dict(l=150, r=60, t=60, b=40))
    fig.show()

    ## 8.1 Grouped Bar Chart - Top 10 VS Bottom 10 - Yearwise - Mean
    # top 10 countries dataframe subsetting
    top_ten = df[df['rank']<=10].sort_values(['year', 'rank']).reset_index(drop=True)
    top_ten['category'] = 'top_ten'

    # bottom 10 countries datafram subsetting
    df_sorted = df.sort_values(['year', 'rank'])
    subset_tail_df = df_sorted.groupby('year').tail(10)
    subset_tail_df = subset_tail_df.reset_index(drop=True)
    subset_tail_df['rank'] = subset_tail_df.groupby(['year']).cumcount() + 1
    subset_tail_df['category'] = 'bottom_ten'

    # union all dataframes
    top_bot = pd.concat([top_ten, subset_tail_df])
    top_bot

    # grouping by
    top_bot_mean = top_bot.groupby(['year', 'category']).agg(score_mean=('score', np.mean),
                                                            gdp_per_capita=('gdp_per_capita', np.mean),
                                                            social_support=('social_support', np.mean),
                                                            life_expectancy=('life_expectancy', np.mean),
                                                            freedom=('freedom', np.mean),
                                                            generosity=('generosity', np.mean),
                                                            corruption=('corruption', np.mean)).reset_index()

    # chart
    def plot_mean_values(data, x_variable, y_variable, title):
        fig = go.Figure()

        x_values = data[x_variable].unique()

        for category in data['category'].unique():
            category_data = data[data['category'] == category]
            fig.add_trace(go.Bar(
                x=x_values,
                y=category_data[y_variable],
                name=category
            ))

        fig.update_layout(
            title=title,
            xaxis=dict(tickmode='linear', tick0=min(x_values), dtick=1),
            yaxis=dict(title=y_variable.capitalize()),
            barmode='group',
            width=800,
            height=500,
            margin=dict(l=50, r=50, b=100, t=100)
        )

        fig.show()

    plot_mean_values(top_bot_mean, 'year', 'score_mean', 'Mean Happiness Score by Year and Category')
    plot_mean_values(top_bot_mean, 'year', 'gdp_per_capita', 'Mean gdp_per_capita by Year and Category')
    plot_mean_values(top_bot_mean, 'year', 'social_support', 'Mean social_support by Year and Category')
    plot_mean_values(top_bot_mean, 'year', 'life_expectancy', 'Mean life_expectancy by Year and Category')
    plot_mean_values(top_bot_mean, 'year', 'freedom', 'Mean freedom by Year and Category')
    plot_mean_values(top_bot_mean, 'year', 'generosity', 'Mean generosity by Year and Category')
    plot_mean_values(top_bot_mean, 'year', 'corruption', 'Mean corruption by Year and Category')

    ## 8.2 Grouped Bar Chart - Yearwise - All Factors
    df_agg = df.groupby(['year']).agg(gdp_per_capita=('gdp_per_capita', np.mean),
                                    social_support=('social_support', np.mean),
                                    life_expectancy=('life_expectancy', np.mean),
                                    freedom=('freedom', np.mean),
                                    generosity=('generosity', np.mean),
                                    corruption=('corruption', np.mean)).reset_index()

    df_agg_T = df_agg.set_index('year').T.reset_index()
    df_agg_T

    fig = go.Figure()

    for year in df_agg_T.columns[1:]:
        fig.add_trace(go.Bar(
            x=df_agg_T['index'],
            y=df_agg_T[year],
            name=str(year),
        ))

    # Update layout
    fig.update_layout(
        title="Grouped Bar Chart - Factors by Year",
        xaxis_title="Factors",
        yaxis_title="Average Value",
        barmode="group",
        legend_title="Year",
        height=600
    )

    # Show the plot
    fig.show()