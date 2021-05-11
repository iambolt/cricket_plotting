import pandas as pd
import random
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import dash_table
import plotly.express as px  
import dash 
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output





app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LUX])
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


def _cal_probab(dataframe_dismissials, dataframe_final):
    
    list_probab = []
    for i in dataframe_dismissials.zones.unique():
        dismis = len(dataframe_dismissials.query(f'zones == "{i}"'))
        tot = len(dataframe_final.query(f'zones== "{i}"'))
        prob = dismis / tot
        list_probab.append({i : prob})
        
    data_dismiss = pd.DataFrame.from_records(list_probab)


    dum = pd.Series(np.diag(data_dismiss), index=[data_dismiss.index, data_dismiss.columns])

    dum_1 = pd.DataFrame(dum, columns=['Probability'])
    dum_1.reset_index(inplace=True)
    dum_1.drop('level_0', axis=1, inplace=True)

    dum_1.rename(columns={"level_1": "Zones"},inplace=True)
    probab_df = dum_1.copy()
    
    return probab_df
        
        
    
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


def plotting_dfs(size):
    ''' This function creates the dataframes required for plotting
        
        Returns EVENTS dataframe (columns = x coordinate ,y coordinate, Runs scored, zone in which the ball was delivered), 
        and POLAR dataframe includes raidus and width used in plotting scoring areas for Barpolar plot
    '''
    

    
   # generating 120 x and y co ordinates of ball position  with the help of random
   # with specific bound range and combining as dataframe
    
    list_y = []
    for i in range(0,size):
        n = random.randint(0,190)
        list_y.append(n)

    list_x = []
    for i in range(0,size):
        n = random.randint(90,260)
        list_x.append(n)


    xy_list = [list_x,list_y]

    xy_df = pd.DataFrame(xy_list).transpose()
    
    xy_df.columns = ['x','y']
    
    # generating event dataframe from predefined values above
    
    event_col = []
    
    for i in range(size):
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

    probab_df = _cal_probab(dismissal_df,final_df)

    
    return final_df, strike_rate_df, runs_df, barpolar_df, dismissal_df, probab_df
    
    


app.layout = dbc.Container(
    [
        html.Header([

                    dbc.Row([
                        dbc.Col([
                            html.H1("Batter's Striking Zone Analytics"),
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
                    html.H5('Select past N matches'),
                    dcc.Slider(
                    id='matches',
                    min=5,
                    max=15,
                    value=5,
                        marks={
                            5: '5',
                            10: '10',
                            15: '15',
                            },included=False),
                ],width=4),

        
            ],justify="center",
        ),



        ]),


        html.Hr(),


        html.Div([
            dbc.Row([


                dbc.Col([
                    dbc.Spinner(
                        dcc.Graph(id='display_striking_points'),
                        color="dark"),
                        ], width=8
                       ),

                dbc.Col([

                    dbc.Spinner(
                        dcc.Graph(id='dsiplay_scoring_areas'),
                        color="dark"),
                        ], width=4  
                       )

        ],justify="start"),


            ]),

        html.Div([
            dbc.Row([

                dbc.Col([
                    dbc.Spinner(
                        dcc.Graph(id='display_zone_SR',style={'height': '80vh'})
                        ),
                        ], width=6,style={'padding-top':'40px'}
                       ),

                dbc.Col([

                    dbc.Spinner(
                        dcc.Graph(id='display_zone_runs',style={'height': '80vh'})
                        ),
                        ], width=6, style={'padding-top':'40px'}  
                       )

        ],justify="start"),


            ]),



        html.Div([
            dbc.Row([

                dbc.Col([
                    html.H5('Dismissal Likelihood Probability')],
                    style={'padding-left': '900px', 'padding-top':'25px'}
                    ),  

        ]),


            ]),



        html.Div([
            dbc.Row([
                dbc.Col([
                    dbc.Spinner(
                        dcc.Graph(id='display_dismissals')
                        ),
                        ], width=6,  
                       ),


                dbc.Col([
                    html.Div(id='team-data')],
                    style={'padding-top': '70px'}
                    ),                




                ],justify="start"),
            ]),



            ],
    fluid=True
)

# app.get_asset_url('pitch.PNG')
@app.callback(
    [Output('display_striking_points', 'figure'),
    Output('dsiplay_scoring_areas', 'figure'),
    Output('display_zone_SR', 'figure'),
    Output('display_zone_runs', 'figure'),
    Output('display_dismissals', 'figure'),
    Output('team-data', 'children'),
    ],
    [
        Input('dd_player', 'value'),
        Input('matches', 'value'),
    ],
    prevent_initial_call=False
)

def display_plot(dd_player,dd_matches):
    
    
    print(dd_player,dd_matches)


    if dd_player == None:
        fig = px.scatter()
        fig.add_annotation(x=2, y=2,font_size=40,font_color='Blue',
            text="Select a Player",
            showarrow=False,
            yshift=10)
        fig2 = px.scatter()
        fig2.add_annotation(x=2, y=2,font_size=20,font_color='Blue',
            text="Select a Player",
            showarrow=False,
            yshift=10)

        fig3 = px.scatter()
        fig3.add_annotation(x=2, y=2,font_size=40,font_color='Blue',
            text="Select a Player",
            showarrow=False,
            yshift=10)
        fig4 = px.scatter()
        fig4.add_annotation(x=2, y=2,font_size=40,font_color='Blue',
            text="Select a Player",
            showarrow=False,
            yshift=10)

        fig5 = px.scatter()
        fig5.add_annotation(x=2, y=2,font_size=40,font_color='Blue',
            text="Select a Player",
            showarrow=False,
            yshift=10)

        data_empty = {}
        return fig,fig2,fig3,fig4,fig5,data_empty



    s = 0
    if dd_matches ==5:
        s =100

    elif dd_matches ==10:
        s =130

    elif dd_matches ==15:
        s =150

    print(s)
    if dd_player == 'RS' or 'VK'or 'SD' or 'KL' or 'SI' or 'RP' or 'RV' or 'HP' or 'PJ':
        f,s,r,p,d, prob = plotting_dfs(s)
        


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
            fillcolor="LightSalmon", opacity=0.3,
            layer="above", line_width=0,
        ),
        fig.add_vrect(
            x0=237, x1=279,
            fillcolor="turquoise", opacity=0.3,
            layer="above", line_width=0,
        ),


        fig.add_vrect(
            x0=167, x1=202,
            fillcolor="LightSeaGreen", opacity=0.3,
            layer="above", line_width=0,
        ),

        fig.add_vrect(
            x0=132, x1=167,
            fillcolor="lightgoldenrodyellow", opacity=0.3,
            layer="above", line_width=0,
        ),

        fig.add_vrect(
            x0=90, x1=132,
            fillcolor="goldenrod", opacity=0.3,
            layer="above", line_width=0,
        ),

        fig.add_annotation(
                x=112,
                y=250,
                xref="x",
                yref="y",
                text="Zone1",
                showarrow=False,
                font=dict(
                    family="Courier New, monospace",
                    size=16,
                    color="#ffffff"
                    ),
                align="center",
                bordercolor="#c7c7c7",
                borderwidth=2,
                borderpad=4,
                bgcolor="#ff7f0e",
                opacity=0.8
                )

        fig.add_annotation(
                x=149,
                y=250,
                xref="x",
                yref="y",
                text="Zone2",
                showarrow=False,
                font=dict(
                    family="Courier New, monospace",
                    size=16,
                    color="#ffffff"
                    ),
                align="center",
                bordercolor="#c7c7c7",
                borderwidth=2,
                borderpad=4,
                bgcolor="#ff7f0e",
                opacity=0.8
                )

        fig.add_annotation(
                x=185,
                y=250,
                xref="x",
                yref="y",
                text="Zone3",
                showarrow=False,
                font=dict(
                    family="Courier New, monospace",
                    size=16,
                    color="#ffffff"
                    ),
                align="center",
                bordercolor="#c7c7c7",
                borderwidth=2,
                borderpad=4,
                bgcolor="#ff7f0e",
                opacity=0.8
                )

        fig.add_annotation(
                x=220,
                y=250,
                xref="x",
                yref="y",
                text="Zone4",
                showarrow=False,
                font=dict(
                    family="Courier New, monospace",
                    size=16,
                    color="#ffffff"
                    ),
                align="center",
                bordercolor="#c7c7c7",
                borderwidth=2,
                borderpad=4,
                bgcolor="#ff7f0e",
                opacity=0.8
                )

        fig.add_annotation(
                x=258,
                y=250,
                xref="x",
                yref="y",
                text="Zone5",
                showarrow=False,
                font=dict(
                    family="Courier New, monospace",
                    size=16,
                    color="#ffffff"
                    ),
                align="center",
                bordercolor="#c7c7c7",
                borderwidth=2,
                borderpad=4,
                bgcolor="#ff7f0e",
                opacity=0.8
                )


        fig.update_layout(template=None,margin=dict(r=0, l=0, t=0, b=0))




        fig2 = go.Figure(go.Barpolar(
            r=list(p.iloc[0]),
            theta=[22.5, 67.5, 112.5, 157.5, 202.5, 247.5, 292.5,337.5],
            width=list(p.iloc[1]),
            marker_color=px.colors.diverging.Earth,
            marker_line_color="black",
            marker_line_width=2,
            opacity=0.8
        ))


        fig2.update_layout(
                template="ggplot2",
                title = 'Scoring areas',
                showlegend = False,
                polar = dict(
                    radialaxis = dict(range=[0, 5], showticklabels=False, ticks=''),
                    angularaxis = dict(showticklabels=False, ticks='')
                )
            )
        
        fig3 = px.bar(s, x='SR', y='Zones',text='Zones',orientation='h',color='SR',color_continuous_scale=px.colors.diverging.Earth,
                template= "ggplot2",title='Zone-Wise Strike rates')
        fig3.update_yaxes(title='Zones', visible=True, showticklabels=False)
        fig3.update_traces( textposition='inside', opacity=0.8, textfont_color='ghostwhite')

        fig4 = px.bar(r, x='Runs', y='Zones',text='Zones',color='Runs',color_continuous_scale=px.colors.diverging.Earth,orientation='h',
                template= 'ggplot2',title='Zone-Wise Runs scored')
        fig4.update_yaxes(title='Zones', visible=True, showticklabels=False)

        fig4.update_traces( textposition='inside', opacity=0.8, textfont_color='ghostwhite')

        fig5 = px.pie(d, values=d.zones.value_counts().values, names=d.zones.unique(),
                color_discrete_sequence=px.colors.diverging.Earth,template= "ggplot2")
        fig5.update_traces(title='Dismissal Zones',textposition='inside', textinfo='percent+label')



        data_note = []

        data_note.append(html.Div(dash_table.DataTable(
        data= prob.to_dict('records'),
                columns= [{'name': x, 'id': x} for x in prob],
                    style_as_list_view=True,
                    editable=False,
                    style_table={
                        'overflowY': 'scroll',
                        'width': '100%',
                        'minWidth': '100%',
                    },
                    style_header={
                            'backgroundColor': '#f8f5f0',
                            'fontWeight': 'bold'
                        },
                    style_cell={
                            'textAlign': 'center',
                            'padding': '8px',
                        },
                )))

    return fig,fig2,fig3,fig4,fig5,data_note 

if __name__ == '__main__':
    app.run_server(debug=True,use_reloader=False)
    