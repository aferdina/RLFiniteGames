# gridworld backend for the dash app
from rlfinitegames.environments.grid_world import GridWorld
import numpy as np 
import plotly.graph_objs as go


def create_matrix(state: list[int], env: GridWorld) -> np.ndarray:
    """ create game information from grid world"""
    matrix = np.zeros(shape=(env.size, env.size))
    matrix[state[0], state[1]] = 1
    matrix[env.bomb_position[0], env.bomb_position[1]] = 2
    matrix[env.goal_position[0], env.goal_position[1]] = 3
    texts = [["" for _ in range(env.size)] for _ in range(env.size)]
    texts[state[0]][state[1]] = "A"
    texts[env.bomb_position[0]][env.bomb_position[1]] = "B"
    texts[env.goal_position[0]][env.goal_position[1]] = "T"
    return matrix, texts

def costum_render(state:list[int], env: GridWorld):

    matrix_plot, texts = create_matrix(state=state, env=env)

   # Define the tick values and labels for the x-axis and y-axis
    x_tickvals = np.arange(env.size)
    x_ticktext = [str(i) for i in x_tickvals]
    y_tickvals = np.arange(env.size)
    y_ticktext = [str(i) for i in y_tickvals]


    # Create a Plotly heatmap object with the matrix and color scale
    heatmap = go.Heatmap(z=matrix_plot, text=texts)

    # Create a Plotly figure object with the heatmap
    fig = go.Figure(data=[heatmap])

    # Set the title of the figure
    fig.update_layout(title='Grid World', height=500, width=500, xaxis={'tickvals': x_tickvals, 'ticktext': x_ticktext}, yaxis={
                    'tickvals': y_tickvals, 'ticktext': y_ticktext, "autorange": 'reversed'})
    # Add text annotations to the heatmap
    for i in range(env.size):
        for j in range(env.size):
            fig.add_annotation(x=j, y=i, text=texts[i][j], showarrow=False, font=dict(
                color='black', size=12), xref='x', yref='y', align='center', valign='middle')

    # Display the figure in a Plotly chart

    return fig