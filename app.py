import base64
import json
import numpy as np
import dash_table
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
from process_BRDF_file import process_BRDF_json
from helper_functions import *
import colour as clr
import plotly.graph_objects as go

#App configurations
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server
app.config['suppress_callback_exceptions'] = True
colors = {'background': '#111111', 'text': '#000080'}

global_data = []

def server_layout():
    layout = html.Div([
        dcc.Store(id='browser_data_storage',storage_type='memory'),
        dcc.Store(id='selected_data',storage_type='memory'),
        dcc.Store(id='selected_phiv',storage_type='memory'),
        dcc.Store(id='tristimulus_XYZ_values',storage_type='memory'),
        dcc.Store(id='RGB_values',storage_type='memory'),
        # html.Div(id='browser_data_storage', style={'display': 'none'}),
        html.H1(children='BiRD view v2.0',
                style={
                    'textAlign': 'center',
                    'color': colors['text']
                }),
        html.Div(children='''A web application for BRDF data visualization.''',
                 style={
                     'textAlign': 'center',
                     'color': colors['text']
                 }),
        html.Hr(),
        html.Div(children=[
            dcc.Upload(
                id='upload-data',
                children=html.Div(['Drag and Drop or ', html.A('Select Files')]),
                style={
                    'width': '100%',
                    'height': '60px',
                    'lineHeight': '60px',
                    'borderWidth': '1px',
                    'borderStyle': 'dashed',
                    'borderRadius': '5px',
                    'textAlign': 'center'
                },
                multiple=True,
            )
        ], className="row ", id='upload-file'),
        html.Hr(style={'margin-bottom':'1px'}),
        html.Div(
            dcc.Tabs(id="menu-tabs", value='menu', children=[
                dcc.Tab(id="applet-tab", label='Applet',children=[
                    html.Div(dcc.Dropdown(id="Wavelength", placeholder="Wavelength"),
                             style={'display': 'inline-block', 'width': '16.6%'}),
                    html.Div(dcc.Dropdown(id="ThetaI", placeholder="Incidence zenith angle"),
                             style={'display': 'inline-block', 'width': '16.6%'}),
                    html.Div(dcc.Dropdown(id="PhiI", placeholder="Incidence azimuthal angle"),
                             style={'display': 'inline-block', 'width': '16.6%'}),
                    html.Div(dcc.Dropdown(id="Polarization", placeholder="Polarization"),
                             style={'display': 'inline-block', 'width': '16.6%'}),
                    html.Div(dcc.Dropdown(id="Observer", placeholder="Observer"),
                             style={'display': 'inline-block', 'width': '16.6%'}),
                    html.Div(dcc.Dropdown(id="Illuminant", placeholder="Illuminant"),
                             style={'display': 'inline-block', 'width': '16.6%'}),

                    dcc.Tabs(id="applet-modes", value='menu-1', children=[
                             dcc.Tab(id="applet-BRDF", label='BRDF visualisation', value='BRDF', children=[
                                  html.Div(dcc.Graph(id="3D-plot"), style={'display': 'inline-block', 'width': '50%'}),
                                 html.Div(dcc.Graph(id="Point-spectrum"), style={'display': 'inline-block', 'width': '50%'}),
                                 html.Div(dcc.Graph(id="Projection-plot"), style={'display': 'inline-block', 'width': '50%'}),
                                 html.Div(dcc.Graph(id="2D-BRDF"), style={'display': 'inline-block', 'width': '50%'})]),
                             dcc.Tab(id="applet-COLOR", label='CIELAB', value='CIELAB', children=[
                                 html.Div(dcc.Graph(id="CIELAB-3Dplot"), style={'display': 'inline-block', 'width': '50%'}),
                                 html.Div(dcc.Graph(id="CIEAB-plot"), style={'display': 'inline-block', 'width': '50%'}),
                                 html.Div(dcc.Graph(id="CIELAB-plot"), style={'display': 'inline-block', 'width': '50%'}),
                                ])]
                             )]
                        ),
                dcc.Tab(id="help-tab", label='Help',value='menu-2', children=html.Div(children='''Here is help text'''))])
        )
    ])
    return layout

@app.callback([Output('browser_data_storage','data'),
               Output('Wavelength','options'),
               Output('ThetaI','options'),
               Output('PhiI','options'),
               Output('Polarization','options'),
               Output('Observer','options'),
               Output('Illuminant','options'),
               Output('Wavelength','value'),
               Output('ThetaI','value'),
               Output('PhiI','value'),
               Output('Polarization','value'),
               Output('Observer','value'),
               Output('Illuminant','value'),
               Output('upload-data', 'filename')],
              [Input('upload-data', 'contents')])
def upload_file(contents):
    if contents is None:
        raise PreventUpdate

    content_type, content_string = contents[0].split(',')
    decoded = base64.b64decode(content_string)
    processed_data = process_BRDF_json(json.loads(decoded.decode('utf-8')))

    u_wls = processed_data['Wavelengths']
    u_theta_Is = processed_data['thetaIs']
    u_phi_Is = processed_data['phiIs']
    u_pols = processed_data['Polarization']

    opt_u_wls = [{'label':value,'value':value} for value in u_wls]
    opt_u_theta_Is = [{'label':value,'value':value} for value in u_theta_Is]
    opt_u_phi_Is = [{'label':value,'value':value} for value in u_phi_Is]
    opt_u_pols = [{'label':value,'value':value} for value in u_pols]
    opt_observer = [{'label':value,'value':value} for value in clr.STANDARD_OBSERVERS_CMFS]
    opt_illuminant = [{'label':value,'value':value} for value in clr.ILLUMINANTS_SDS]

    opt_u_wls_init = u_wls[0]
    opt_u_theta_Is_init = u_theta_Is[0]
    opt_u_phi_Is_init = u_phi_Is[0]
    opt_u_pols_init = u_pols[0]
    opt_observer_init = 'CIE 1931 2 Degree Standard Observer'
    optilluminant_init = 'D65'

    return processed_data, opt_u_wls, opt_u_theta_Is, opt_u_phi_Is, opt_u_pols, opt_observer, opt_illuminant, opt_u_wls_init, opt_u_theta_Is_init, opt_u_phi_Is_init, opt_u_pols_init, opt_observer_init, optilluminant_init, None

@app.callback(Output('upload-data', 'style'),
              [Input('upload-data', 'filename')],
              [State('upload-data', 'style')])
def rise_processing_state(filename,style):
    if filename is None:
        style['backgroundColor'] = '#FFFFFF'
    else:
        style['backgroundColor'] = '#FE4A4C'
    return style

@app.callback([Output('selected_data','data'),
               Output('tristimulus_XYZ_values','data'),
               Output('RGB_values','data'),
               Output('3D-plot','figure')],
              [Input('Wavelength','value'),
               Input('ThetaI','value'),
               Input('PhiI','value'),
               Input('Polarization','value'),
               Input('Observer','value'),
               Input('Illuminant','value')],
              [State('browser_data_storage','data')])
def update_3D_plot(wl, thI, phiI, pol, observer, illuminant, data):
    if wl is None or thI is None or phiI is None or pol is None or observer is None or illuminant is None:
        # print('no data')
        raise PreventUpdate

    theta = np.array(data['thetaVs'])[np.newaxis]
    phi = np.array(data['phiVs'])[np.newaxis]
    X = theta.T*np.cos(np.radians(phi))
    Y = theta.T*np.sin(np.radians(phi))
    Z = select_data(wl, thI, phiI, pol, data)

    tristimulus_values, RGB_values = get_tristimulus_XYZs(thI,phiI,pol,data,observer,illuminant)

    # print(tristimulus_values)
    # print(X.shape,Y.shape,Z.shape)

    figure = go.Figure()
    figure.add_trace(go.Surface(x=X, y=Y, z=Z))

    figure.update_layout(title="BRDF 3D plot",
        scene=dict(
            xaxis_title="Theta (deg)",
            yaxis_title="Theta (deg)",
            zaxis_title="Radiance factor"
        )
    )

    return Z, tristimulus_values, RGB_values, figure

@app.callback(Output('Projection-plot','figure'),
              [Input('3D-plot','figure')],
              [State('browser_data_storage','data'),
               State('RGB_values','data')])
def update_projection_plot(figure, data, RGB_values):
    if figure is None or data is None or RGB_values is None:
        # print('no data')
        raise PreventUpdate

    thetas = np.array(data['thetaVs'])
    arranged_thetas = np.append(np.flip(thetas[thetas<0]),thetas[thetas>=0])
    phis = np.array(data['phiVs'])
    RGB_values = np.array(RGB_values)

    figure = go.Figure()

    previous_radius = 0
    for theta in arranged_thetas:
        scale = np.max(np.abs(arranged_thetas))
        radius = 2*np.sin(np.radians(theta)/2)
        delta_r = np.abs(scale*(previous_radius-radius)/np.sqrt(2))
        for phi in phis:
            RGB_color = RGB_values[thetas == theta, phis == phi][0]
            # print(RGB_color)
            if theta == 0:
                delta_r = 0
            if theta >= 0:
                figure.add_trace(go.Barpolar(
                    name = str(theta),
                    r=[delta_r],
                    theta=[phi],
                    marker_color='rgb('+str(RGB_color[0])+','+str(RGB_color[1])+','+str(RGB_color[2])+')'))
            elif theta < 0:
                figure.add_trace(go.Barpolar(
                    name=str(theta),
                    r=[delta_r],
                    theta=[180+phi],
                    marker_color='rgb('+str(RGB_color[0])+','+str(RGB_color[1])+','+str(RGB_color[2])+')'))
        previous_radius = radius

        figure.update_layout(
            title="BRDF polar representation"
        )

            # if theta < 0:
            #     figure.add_trace(pgo.Barpolar(
            #     r=np.flip(np.array([2*np.cos(np.radians(theta)) for angles in thetas])),
            #     theta=180+phis))
    return figure

@app.callback([Output('2D-BRDF','figure'),
               Output('selected_phiv','data')],
              [Input('Projection-plot','figure'),
               Input('Projection-plot','relayoutData')],
              [State('browser_data_storage','data'),
               State('selected_data','data')])
def update_2D_brdf_plot(fig, relayoutData, data, selected_data):
    if fig is None:
        raise PreventUpdate

    figure = go.Figure()
    phis = np.array(data['phiVs'])
    thetas = np.array(data['thetaVs'])
    data = np.array(data['data'])
    selected_data = np.array(selected_data)
    relayoutData = relayoutData

    if relayoutData is None:
        relayoutData = {'polar.angularaxis.rotation': 0}
    if not 'polar.angularaxis.rotation' in relayoutData:
        relayoutData['polar.angularaxis.rotation'] = 0

    selected_angle = [0]
    if 'polar.angularaxis.rotation' in relayoutData:
        angle = relayoutData['polar.angularaxis.rotation']
        if np.abs(angle) > 180:
            raise PreventUpdate
        else:
            if angle > 0:
                angle = np.abs(angle)
                d = np.abs(phis-angle)
                min_d = np.min(d)
                selected_angle = 360-phis[d == min_d]
            elif angle <= 0:
                angle = np.abs(angle)
                d = np.abs(phis-angle)
                min_d = np.min(d)
                selected_angle = phis[d == min_d]
    else:
        raise PreventUpdate

    selected_angle = selected_angle[0]
    if selected_angle == 180:
        phi_mask = np.logical_or(phis == 180, phis == 0)
    elif selected_angle == 360:
        phi_mask = np.logical_or(phis == 0,phis == 180)
    elif selected_angle < 180:
        phi_mask = np.logical_or(phis == selected_angle, phis == (selected_angle+180))
    elif selected_angle > 180:
        phi_mask = np.logical_or(phis == selected_angle, phis == (selected_angle-180))

    x = thetas
    y = selected_data[:,phi_mask].T

    selected_phiVs = phis[phi_mask]
    print(selected_phiVs)

    if phis[phi_mask].shape[0] == 1:
        figure.add_trace(go.Scatter(name='BRDF',x = x, y = y[0], mode='lines+markers'))
    elif phis[phi_mask].shape[0] == 2:
        figure.add_trace(go.Scatter(name='BRDF 0 to 90',x = x, y = y[0], mode='lines+markers'))
        figure.add_trace(go.Scatter(name='BRDF -90 to 0',x = -x, y = y[1], mode='lines+markers'))
    else:
        raise PreventUpdate

    figure.update_layout(
        title="BRDF 2D plot at selected viewing azimuth",
        xaxis_title='Viewing zenith angle Theta (deg)',
        yaxis_title='Radiance factor'
    )

    return figure, selected_phiVs

@app.callback(Output('Point-spectrum','figure'),
              [Input('2D-BRDF','clickData'),
               Input('2D-BRDF','figure')],
              [State('browser_data_storage','data'),
               State('ThetaI','value'),
               State('PhiI','value'),
               State('Polarization','value'),
               State('selected_phiv','data')])
def update_Spectrum_plot(clickData, fig, data, ThetaI, PhiI, Polarization, selected_phiVs):
    if fig is None:
        raise PreventUpdate

    thetaVs = np.array(data['thetaVs'])
    phiVs = np.array(data['phiVs'])
    selected_phiVs = np.array(selected_phiVs)
    if clickData is not None:
        selected_theta = clickData['points'][0]['x']
    else:
        selected_theta = None

    x = np.array(data['Wavelengths'])
    y = np.empty(1)
    if selected_theta is None:
        y = select_spectrum(thetaVs[0], phiVs[0], ThetaI, PhiI, Polarization, data)
    else:
        if selected_phiVs.shape[0] == 1:
            y = select_spectrum(selected_theta,selected_phiVs[0], ThetaI, PhiI, Polarization, data)
        elif selected_phiVs.shape[0] == 2:
            if clickData['points'][0]['curveNumber'] == 0:
                y = select_spectrum(selected_theta,selected_phiVs[0], ThetaI, PhiI, Polarization, data)
            elif clickData['points'][0]['curveNumber'] == 1:
                y = select_spectrum(-selected_theta,selected_phiVs[1], ThetaI, PhiI, Polarization, data)
    if y.shape[0] == 0:
        y = select_spectrum(thetaVs[0], phiVs[0], ThetaI, PhiI, Polarization, data)

    figure = go.Figure()
    figure.add_trace(go.Scatter(x = x, y = y[0], mode='lines+markers'))

    figure.update_layout(
        title="Reflectance spectr at selected viewing zenith and azimuth",
        xaxis_title='Wavelength (nm)',
        yaxis_title='Radiance factor'
    )
    return figure

@app.callback(Output('CIELAB-3Dplot','figure'),
              [Input('Point-spectrum','figure')],
              [State('tristimulus_XYZ_values','data'),
               State('selected_phiv','data'),
               State('browser_data_storage','data')])
def update_CIELAB_3Dplot(fig, tristimulus_XYZ, selected_phi, data):
    if fig is None or tristimulus_XYZ is None or selected_phi is None:
        raise PreventUpdate

    selected_phi = np.array(selected_phi)
    phiVs = np.array(data['phiVs'])
    tristimulus_XYZ = np.array(tristimulus_XYZ)

    figure = go.Figure()

    if selected_phi.shape[0] == 1:
        selected_XYZ = tristimulus_XYZ[:, phiVs == selected_phi[0]][:,0,:]
        selected_LAB = np.array([clr.XYZ_to_Lab(selected_XYZ[i]/100) for i in range(selected_XYZ.shape[0])])
        figure.add_trace(go.Scatter3d(x=selected_LAB[:,1], y=selected_LAB[:,2], z=selected_LAB[:,0], mode='lines+markers'))
    elif selected_phi.shape[0] == 2:
        selected_XYZ_pos = tristimulus_XYZ[:, phiVs == selected_phi[0]][:,0,:]
        selected_LAB_pos = np.array([clr.XYZ_to_Lab(selected_XYZ_pos[i]/100) for i in range(selected_XYZ_pos.shape[0])])
        figure.add_trace(go.Scatter3d(x=selected_LAB_pos[:, 1], y=selected_LAB_pos[:, 2], z=selected_LAB_pos[:, 0], mode='lines+markers'))
        selected_XYZ_neg = tristimulus_XYZ[:, phiVs == selected_phi[1]][:,0,:]
        selected_XYZ_neg = np.array([clr.XYZ_to_Lab(selected_XYZ_neg[i]/100) for i in range(selected_XYZ_neg.shape[0])])
        figure.add_trace(go.Scatter3d(x=selected_XYZ_neg[:, 1], y=selected_XYZ_neg[:, 2], z=selected_XYZ_neg[:, 0], mode='lines+markers'))
    else:
        raise PreventUpdate

    figure.update_layout(title="Color travel 3D plot in CIELab space",
                         scene=dict(
                             xaxis_title="a*",
                             yaxis_title="b*",
                             zaxis_title="L*"
                         )
                    )

    return figure

@app.callback(Output('CIEAB-plot','figure'),
              [Input('CIELAB-3Dplot','figure')],
              [State('tristimulus_XYZ_values','data'),
               State('selected_phiv','data'),
               State('browser_data_storage','data')])
def update_CIELAB_3Dplot(fig, tristimulus_XYZ, selected_phi, data):
    if fig is None or tristimulus_XYZ is None or selected_phi is None:
        raise PreventUpdate

    selected_phi = np.array(selected_phi)
    phiVs = np.array(data['phiVs'])
    tristimulus_XYZ = np.array(tristimulus_XYZ)

    figure = go.Figure()

    if selected_phi.shape[0] == 1:
        selected_XYZ = tristimulus_XYZ[:, phiVs == selected_phi[0]][:,0,:]
        selected_LAB = np.array([clr.XYZ_to_Lab(selected_XYZ[i]/100) for i in range(selected_XYZ.shape[0])])
        figure.add_trace(go.Scatter(name='projection',x=selected_LAB[:,1], y=selected_LAB[:,2], mode='lines+markers'))
    elif selected_phi.shape[0] == 2:
        selected_XYZ_pos = tristimulus_XYZ[:, phiVs == selected_phi[0]][:,0,:]
        selected_LAB_pos = np.array([clr.XYZ_to_Lab(selected_XYZ_pos[i]/100) for i in range(selected_XYZ_pos.shape[0])])
        figure.add_trace(go.Scatter(name='projection 0 to 90',x=selected_LAB_pos[:, 1], y=selected_LAB_pos[:, 2], mode='lines+markers'))
        selected_XYZ_neg = tristimulus_XYZ[:, phiVs == selected_phi[1]][:,0,:]
        selected_XYZ_neg = np.array([clr.XYZ_to_Lab(selected_XYZ_neg[i]/100) for i in range(selected_XYZ_neg.shape[0])])
        figure.add_trace(go.Scatter(name='projection -90 to 0',x=selected_XYZ_neg[:, 1], y=selected_XYZ_neg[:, 2], mode='lines+markers'))
    else:
        raise PreventUpdate

    figure.update_layout(
        title="CIELab colorspace projection to a*b* plane",
        xaxis_title='a*',
        yaxis_title='b*'
    )

    return figure

@app.callback(Output('CIELAB-plot','figure'),
              [Input('CIEAB-plot','figure')],
              [State('tristimulus_XYZ_values','data'),
               State('selected_phiv','data'),
               State('browser_data_storage','data')])
def update_CIELAB_3Dplot(fig, tristimulus_XYZ, selected_phi, data):
    if fig is None or tristimulus_XYZ is None or selected_phi is None:
        raise PreventUpdate

    selected_phi = np.array(selected_phi)
    phiVs = np.array(data['phiVs'])
    thetaVs = np.array(data['thetaVs'])
    tristimulus_XYZ = np.array(tristimulus_XYZ)

    figure = go.Figure()

    if selected_phi.shape[0] == 1:
        selected_XYZ = tristimulus_XYZ[:, phiVs == selected_phi[0]][:,0,:]
        selected_LAB = np.array([clr.XYZ_to_Lab(selected_XYZ[i]/100) for i in range(selected_XYZ.shape[0])])
        figure.add_trace(go.Scatter(name='L*',y=selected_LAB[:,0], x=thetaVs, mode='lines+markers',marker=dict(color='yellow'),line=dict(color='yellow')))
        figure.add_trace(go.Scatter(name='a*',y=selected_LAB[:, 1], x=thetaVs, mode='lines+markers',marker=dict(color='red'),line=dict(color='red')))
        figure.add_trace(go.Scatter(name='b*',y=selected_LAB[:, 2], x=thetaVs, mode='lines+markers',marker=dict(color='blue'),line=dict(color='blue')))
    elif selected_phi.shape[0] == 2:
        selected_XYZ_pos = tristimulus_XYZ[:, phiVs == selected_phi[0]][:,0,:]
        selected_LAB_pos = np.array([clr.XYZ_to_Lab(selected_XYZ_pos[i]/100) for i in range(selected_XYZ_pos.shape[0])])
        figure.add_trace(go.Scatter(name='L* 0 to 90',y=selected_LAB_pos[:, 0], x=thetaVs, mode='lines+markers', marker=dict(color='yellow'),line=dict(color='yellow')))
        figure.add_trace(go.Scatter(name='a* 0 to 90',y=selected_LAB_pos[:, 1], x=thetaVs, mode='lines+markers', marker=dict(color='red'),line=dict(color='red')))
        figure.add_trace(go.Scatter(name='b* 0 to 90',y=selected_LAB_pos[:, 2], x=thetaVs, mode='lines+markers', marker=dict(color='blue'),line=dict(color='blue')))
        selected_XYZ_neg = tristimulus_XYZ[:, phiVs == selected_phi[1]][:,0,:]
        selected_XYZ_neg = np.array([clr.XYZ_to_Lab(selected_XYZ_neg[i]/100) for i in range(selected_XYZ_neg.shape[0])])
        figure.add_trace(go.Scatter(name='L* -90 to 0',y=selected_XYZ_neg[:, 0], x=-thetaVs, mode='lines+markers',marker=dict(color='yellow'),line=dict(color='yellow')))
        figure.add_trace(go.Scatter(name='a* -90 to 0',y=selected_XYZ_neg[:, 1], x=-thetaVs, mode='lines+markers',marker=dict(color='red'),line=dict(color='red')))
        figure.add_trace(go.Scatter(name='b* -90 to 0',y=selected_XYZ_neg[:, 2], x=-thetaVs, mode='lines+markers',marker=dict(color='blue'),line=dict(color='blue')))
    else:
        raise PreventUpdate

    figure.update_layout(
        title="CIELab values dependence on viewing zenith angle",
        xaxis_title='CIELab units',
        yaxis_title='Viewing zenith angle Theta (deg)'
    )

    return figure

app.layout = server_layout()

if __name__ == '__main__':
    app.run_server(debug=True)
