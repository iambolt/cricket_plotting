import pandas as pd
import random
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px  
import dash 
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output





app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SLATE])
server = app.server






# constants used
# using colors Red : Dots , white : wicket, goldenrod : boundaries and blue : runs scored
# with a probability distribution of occerence
    
events_color = ['white','red','goldenrod', 'blue']
events_distribution = [0.075,.525,0.25,0.15]


# boundaries with probabilty distribution 

boundaries = [4, 6]
bdistribution = [.8, .2]

# 1's 2's 3's with probabilty distribution 

runs = [1,2,3]
rdistribution = [0.5,.35,.15]


def _cal_counts_dismissals(dataframe):
    ''' identifies the zones where the dismissal occured and 
        
        returns value_counts of the same
    '''
    
    dismissal_df = dataframe.query('events == "white"')
    
    return dismissal_df


def _cal_zone_runs(dataframe):
    
    '''  calculates the runs for the 5 zones
    
        returns dataframe containing zones wise runs
    '''
    
    runs_list = []

    for i in dataframe.zones.unique():
        new_df = dataframe.query(f'zones == "{i}"')
        runs = {f'{i}' : new_df.Runs.sum()}
        runs_list.append(runs.copy())

    runs_df = pd.DataFrame.from_records(runs_list)

    dum = pd.Series(np.diag(runs_df), index=[runs_df.index, runs_df.columns])

    dum_1 = pd.DataFrame(dum, columns=['Runs'])
    dum_1.reset_index(inplace=True)
    dum_1.drop('level_0', axis=1, inplace=True)
    dum_1.rename(columns={"level_1": "Zones"},inplace=True)

    runs_df = dum_1.copy()
    
    return runs_df
    
    
def _cal_zone_strike_rate(dataframe): 
    '''  calculates the strike rate for the 5 zones
    
        returns dataframe containing zones wise Strike rate
    '''
    sr_list = []
    for i in dataframe.zones.unique():
        new_df = dataframe.query(f'zones == "{i}"')
        sr = {f'{i}' : (new_df.Runs.sum()/ len(new_df))*100}
        sr_list.append(sr.copy())
        
    sr_df = pd.DataFrame.from_records(sr_list)
        
    dum = pd.Series(np.diag(sr_df), index=[sr_df.index, sr_df.columns])
        
    dum_1 = pd.DataFrame(dum, columns=['SR'])
    dum_1.reset_index(inplace=True)
    dum_1.drop('level_0', axis=1, inplace=True)
    
    dum_1.rename(columns={"level_1": "Zones"},inplace=True)
    
    strike_rate_df = dum_1.copy()
    
    return strike_rate_df


        
    
def _zone_finder(dataframe):
    ''' helper funtion for determining the zone in
        which the balls was delivererd
        
        return a list 'zones'
    '''
    zones = []
    for x in dataframe.x:
        if x>=90 and x<132:
            zones.append('Zone 1')
            
        if x>=132 and x<167:
            zones.append('Zone 2')

        if x>=167 and x<202:
            zones.append('Zone 3')
            
        if x>=202 and x<237:
            zones.append('Zone 4')

        if x>=237 and x<279:
            zones.append('Zone 5')
                        
    return zones


def plotting_dfs():
    ''' This function creates the dataframes required for plotting
        
        Returns EVENTS dataframe (columns = x coordinate ,y coordinate, Runs scored, zone in which the ball was delivered), 
        and POLAR dataframe includes raidus and width used in plotting scoring areas for Barpolar plot
    '''
    

    
   # generating 120 x and y co ordinates of ball position  with the help of random
   # with specific bound range and combining as dataframe
    
    list_y = []
    for i in range(0,120):
        n = random.randint(0,190)
        list_y.append(n)

    list_x = []
    for i in range(0,120):
        n = random.randint(90,260)
        list_x.append(n)


    xy_list = [list_x,list_y]

    xy_df = pd.DataFrame(xy_list).transpose()
    
    xy_df.columns = ['x','y']
    
    # generating event dataframe from predefined values above
    
    event_col = []
    
    for i in range(120):
        random_event = random.choices(events_color, events_distribution)
        event_col.append(random_event[0])
    
    
    new_cols = pd.DataFrame(event_col, columns =['events'])
    
    new_cols['Runs'] = [0 if x=='white' else 0 if x=="red" else random.choices(boundaries, bdistribution)[0] if x=='goldenrod' else random.choices(runs, rdistribution)[0] if x=='blue' else "" for x in new_cols['events']]
    
    xy_events_df = pd.concat([xy_df, new_cols], axis=1)
    
    
    # determining the zone of the ball placed with the help of x co-ordinate value
    
    zon = _zone_finder(xy_events_df)
    
    xy_events_df['zones'] = zon
    
    final_df = xy_events_df.copy()
    
    
    strike_rate_df = _cal_zone_strike_rate(final_df)
    
    runs_df = _cal_zone_runs(final_df)
    
    
    # generating raidus and width for scoring area Barpolar graph
    
    polar_r = []
    for i in range(0,8):
        n = random.uniform(2,4.8)
        polar_r.append(round(n,2))


    polar_width = []
    for i in range(0,8):
        n = random.randint(15,45)
        polar_width.append(n)

    
    barpolar_df = pd.DataFrame([polar_r,polar_width])
    
    
    dismissal_df = _cal_counts_dismissals(final_df)
    
    return final_df, strike_rate_df, runs_df, barpolar_df, dismissal_df
    
    



top_card = dbc.Card(
    [
        dbc.CardImg(src=app.get_asset_url('ben.jpg'), top=True),
        dbc.CardBody(
            html.P("Kohli", className="card-text")
        ),
    ],
    
)



app.layout = dbc.Container(
    [
        html.Header([

                    dbc.Row([
                        dbc.Col([
                            html.H1("Batter Performance Analytics"),
                            html.H5(html.Em("Nishant Singh Siddhu")),         
                        ],style={'text-align': 'center'})    
                        ]),
        ]),

        html.Hr(),

        html.Div([
            dbc.Row(
            [
                dbc.Col([
                    html.H5('Select Player'),
                    dcc.Dropdown(
                        id='dd_player',
                        options=[
                            {'label': 'Virat Kohli', 'value': 'VK'},
                            {'label': 'Rohit Sharma', 'value': 'RS'},
                            {'label': 'Shikhar Dhawan', 'value': 'SD'},
                            {'label': 'KL Rahul', 'value': 'KL'},
                            {'label': 'Shreyas Iyer', 'value': 'SI'},
                            {'label': 'Rishab Pant', 'value': 'RP'},
                            {'label': 'Ravi Jadega', 'value': 'RV'},
                            {'label': 'Hardik', 'value': 'HP'},
                            {'label': 'Pujara', 'value': 'PJ'}
                        ],
                        searchable=False
                    ),
                ],width=4),


                dbc.Col([
                    html.H5('Select last N matches'),
                    dcc.Dropdown(
                        id='dd_matches',
                        options=[
                            {'label': 'Last 5 matches', 'value': '5'},
                            {'label': 'Last 7 matches', 'value': '7'},
                            {'label': 'Last 10 matches', 'value': '10'}
                        ],
                        searchable=False
                    ),
                ],width=4),


                dbc.Col([
                    dbc.Button("Analyze", id="analyze_btn", color="success",
                               n_clicks='0',size="lg")
                ],width=4, style={'padding-top':'25px'})
        
            ],justify="start",
        ),



        ]),


        html.Hr(),


        html.Div([
            dbc.Row([
                dbc.Col([dbc.Card(top_card)],
                width=2
                       ),

                dbc.Col([
                    dbc.Spinner(
                        dcc.Graph(id='display'),
                        color="dark"),
                        ], width=5
                       ),

                dbc.Col([

                    dbc.Spinner(
                        dcc.Graph(id='dis'),
                        color="dark"),
                        ], width=5  
                       )

        ],justify="start"),


            ])
        

    
    ],
    fluid=True
)


@app.callback(
    [Output('display', 'figure'),
    Output('dis', 'figure')],
    [
        Input('dd_player', 'value'),
        Input('dd_matches', 'value'),
        Input('analyze_btn', 'n_clicks'),
    ],
    prevent_initial_call=True
)

def display_plot(dd_player,analyze_btn,dd_matches):
    
    print(dd_player,analyze_btn,dd_matches)
    if dd_player == 'RS':
        f,s,r,p,d = plotting_dfs()
        


        fig = go.Figure()
        # Add image
        img_width = 432
        img_height = 264
        fig.add_layout_image(
                x=0,
                sizex=img_width,
                y=0,
                sizey=img_height,
                xref="x",
                yref="y",
                opacity=1.0,
                layer="below",
                source=app.get_asset_url('pitch.PNG')
        )
        fig.update_xaxes(visible=False,showgrid=False, range=(0, img_width))
        fig.update_yaxes(visible=False,showgrid=False,scaleanchor='x', range=(img_height, 0))

        fig.add_trace(go.Scatter(x=f.x,
                                y=f.y,
                                mode='markers',
                                text=f['Runs'],
                                marker=dict(size=20,color=f.events.to_list())))






        fig.add_vrect(
            x0=202, x1=237,
            fillcolor="LightSalmon", opacity=0.23,
            layer="above", line_width=0,
        ),
        fig.add_vrect(
            x0=237, x1=279,
            fillcolor="turquoise", opacity=0.23,
            layer="above", line_width=0,
        ),


        fig.add_vrect(
            x0=167, x1=202,
            fillcolor="LightSeaGreen", opacity=0.23,
            layer="above", line_width=0,
        ),

        fig.add_vrect(
            x0=132, x1=167,
            fillcolor="lightgoldenrodyellow", opacity=0.23,
            layer="above", line_width=0,
        ),

        fig.add_vrect(
            x0=90, x1=132,
            fillcolor="goldenrod", opacity=0.23,
            layer="above", line_width=0,
        ),




        fig.update_layout(title= "SSup",template="plotly_dark",margin=dict(r=0, l=0, t=0, b=0))

        #visible=False,    PaleTurquoise
        fig.show(config={'doubleClick': 'reset'})    





        fig2 = go.Figure(go.Barpolar(
            r=list(p.iloc[0]),
            theta=[22.5, 67.5, 112.5, 157.5, 202.5, 247.5, 292.5,337.5],
            width=list(p.iloc[1]),
            marker_color=["#E4FF87", '#709BFF', '#EF553B', '#FFA15A', '#AB63FA', '#FF6692', '#B6FFB4','#B6E880'],
            marker_line_color="black",
            marker_line_width=2,
            opacity=0.8
        ))


        fig2.update_layout(
                template="plotly_dark",
                title = 'Scoring areas',
                showlegend = False,
                polar = dict(
                    radialaxis = dict(range=[0, 5], showticklabels=False, ticks=''),
                    angularaxis = dict(showticklabels=False, ticks='')
                )
            )
        




    return fig,fig2    


if __name__ == '__main__':
    app.run_server(debug=True,use_reloader=False)
  