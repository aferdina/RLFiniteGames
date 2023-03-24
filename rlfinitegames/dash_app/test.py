import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State

app = dash.Dash(__name__)

app.layout = html.Div([
    dcc.Input(id='input', type='text', value=''),
    html.Div(id='output')
])

@app.callback(
    Output('output', 'children'),
    [Input('input', 'value')],
    [State('input', 'id')]
)
def process_keydown_event(value, input_id):
    if input_id == 'input':
        if 'ArrowUp' in value:

            return 'Up arrow key pressed'
        elif 'ArrowDown' in value:
            return 'Down arrow key pressed'
        elif 'ArrowLeft' in value:
            return 'Left arrow key pressed'
        elif 'ArrowRight' in value:
            return 'Right arrow key pressed'
    return ''

if __name__ == '__main__':
    app.run_server(debug=True)
