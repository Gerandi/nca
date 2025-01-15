import scipy
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
import statsmodels.api as sm
import os
import matplotlib.pyplot as plt
from typing import Dict, Any, Tuple
from scipy.interpolate import interp1d

st.set_page_config(page_title="NCA & Meta-Analysis", layout="wide")

######################################################
# Refined Helper Functions (from previous steps)
######################################################

def check_data_validity(x, y):
    if len(x) < 30:
        raise ValueError("N<30. Not enough data.")
    if np.all(x == x[0]) or np.all(y == y[0]):
        raise ValueError("X or Y is constant.")
    if np.any(np.isnan(x)) or np.any(np.isnan(y)):
        raise ValueError("NaN values found.")
    if np.any(np.isinf(x)) or np.any(np.isinf(y)):
        raise ValueError("Infinite values found.")
    if len(np.unique(x)) < 5:
        raise ValueError("X has too few unique values.")
    return True

def define_scope_limits(x, y):
    Xmin, Xmax = np.min(x), np.max(x)
    Ymin, Ymax = np.min(y), np.max(y)
    Scope = (Xmax - Xmin) * (Ymax - Ymin)
    return Xmin, Xmax, Ymin, Ymax, Scope

def create_sorted_array(x, y):
    arr = np.column_stack((x, y))
    arr = arr[arr[:,0].argsort(kind='mergesort')]
    arr = arr[np.lexsort((arr[:,1], arr[:,0]))]
    return arr

def CE_FDH_envelope_list(sorted_array):
    envelope = [sorted_array[0]]
    current_y = sorted_array[0][1]
    current_x = sorted_array[0][0]
    for i in range(1, len(sorted_array)):
        x_val, y_val = sorted_array[i]
        if x_val == current_x:
            if y_val > current_y:
                current_y = y_val
                envelope[-1] = [x_val, y_val]
        else:
            if y_val >= current_y:
                envelope.append([x_val, current_y])
                current_y = y_val
                current_x = x_val
                envelope.append([x_val, y_val])
            elif i == len(sorted_array)-1:
                envelope.append([x_val, current_y])
    return np.array(envelope)

def CE_FDH_peers(envelope):
    peers = []
    for i in range(1, len(envelope)):
        if envelope[i][0] > envelope[i-1][0]:
            peers.append(envelope[i-1])
        elif i == (len(envelope)-1) and envelope[i][1] > envelope[i-1][1]:
            peers.append(envelope[i])
    return np.array(peers) if len(peers) > 0 else np.array([envelope[0]])

def CE_FDH_effect_size(Xmin, Xmax, Ymin, Ymax, upper_left_edges):
    Scope = (Xmax - Xmin)*(Ymax - Ymin)
    ule = upper_left_edges.copy()
    if ule[-1,0] < Xmax:
        ule = np.vstack([ule, [Xmax, Ymax]])
    area_below_cl = 0
    for i in range(1, len(ule)):
        dx = ule[i,0] - ule[i-1,0]
        dy = ule[i-1,1] - Ymin
        area_below_cl += dx * dy
    es = (Scope - area_below_cl)/Scope
    return np.clip(es, 0, 1)

def OLS_params_from_peers(peers):
    x_peers = peers[:,0]
    y_peers = peers[:,1]
    slope, intercept = np.polyfit(x_peers, y_peers, 1)
    return slope, intercept

def polygon_area(x_coords, y_coords):
    return 0.5*np.abs(np.dot(x_coords, np.roll(y_coords,1)) - np.dot(y_coords, np.roll(x_coords,1)))

def CR_FDH_effect_size(Xmin, Xmax, Ymin, Ymax, slope, intercept):
    Scope = (Xmax - Xmin)*(Ymax - Ymin)
    m = slope
    b = intercept
    y1 = m*Xmin + b
    y2 = m*Xmax + b
    poly = [[Xmax, Ymin]]
    if y2 > Ymax:
        X_intersect = (Ymax - b)/m
        poly.append([Xmax, Ymax])
        poly.append([X_intersect, Ymax])
    else:
        poly.append([Xmax, y2])
    if y1 > Ymin:
        poly.append([Xmin, y1])
        poly.append([Xmin, Ymin])
    else:
        X_intersect_ymin = (Ymin - b)/m
        poly.append([X_intersect_ymin, Ymin])
    poly = np.array(poly)
    area_below_cr = polygon_area(poly[:,0], poly[:,1])
    es = (Scope - area_below_cr)/Scope
    return np.clip(es, 0, 1)

def accuracy_from_line(data_array, slope, intercept):
    above_count = np.sum(data_array[:,1] > (intercept + slope*data_array[:,0]))
    accuracy = 1 - above_count/len(data_array)
    return accuracy

def style_plots():
    plt.rcParams.update({
        'figure.figsize': (10, 8),
        'font.family': 'serif',
        'font.size': 10,
        'axes.titlesize': 10,
        'axes.grid': False,
        'lines.linewidth': 1
    })

def create_nca_plot(x: np.ndarray, y: np.ndarray,
                   ce_envelope: np.ndarray, ce_peers: np.ndarray,
                   cr_slope: float, cr_intercept: float,
                   x_name: str = "X", y_name: str = "Y",
                   show_ols: bool = True) -> go.Figure:
    """Create NCA plot with plotly"""
    Xmin, Xmax = np.min(x), np.max(x)
    Ymin, Ymax = np.min(y), np.max(y)
    margin_size = 0.03

    fig = go.Figure()

    # Draw scope lines
    for val, line_name in [(Xmin, 'Xmin'), (Xmax, 'Xmax'), (Ymin, 'Ymin'), (Ymax, 'Ymax')]:
        if line_name.startswith('X'):
            fig.add_vline(x=val, line_dash="dash", line_color="gray", opacity=0.5)
        else:
            fig.add_hline(y=val, line_dash="dash", line_color="gray", opacity=0.5)

    # Plot observations (data points)
    fig.add_trace(
        go.Scatter(x=x, y=y,
                  mode='markers',
                  marker=dict(color='blue', symbol='star', size=8),
                  name='obs')
    )

    # Plot peer points with both peer and observation markers
    peer_x = ce_peers[:, 0]
    peer_y = ce_peers[:, 1]
    for px, py in zip(peer_x, peer_y):
        # Add peer marker (red circle)
        fig.add_trace(
            go.Scatter(x=[px], y=[py],
                      mode='markers',
                      marker=dict(color='red', size=10),
                      name='peer',
                      showlegend=False)
        )
        # Add observation marker (blue star) for the same point
        fig.add_trace(
            go.Scatter(x=[px], y=[py],
                      mode='markers',
                      marker=dict(color='blue', symbol='star', size=8),
                      name='obs',
                      showlegend=False)
        )

    # Plot CE-FDH line
    fig.add_trace(
        go.Scatter(x=ce_envelope[:, 0], y=ce_envelope[:, 1],
                  mode='lines',
                  line=dict(color='red', dash='dash'),
                  name='ce_fdh')
    )

    # Plot CR-FDH line if available
    if not np.isnan(cr_slope):
        x_range = np.linspace(Xmin, Xmax, 100)
        cr_y = cr_intercept + cr_slope * x_range
        fig.add_trace(
            go.Scatter(x=x_range, y=cr_y,
                      mode='lines',
                      line=dict(color='orange', dash='dot'),
                      name='cr_fdh')
        )

    # Plot OLS line if requested
    if show_ols:
        x_range = np.linspace(Xmin, Xmax, 100)
        ols_slope, ols_intercept = np.polyfit(x, y, 1)
        fig.add_trace(
            go.Scatter(x=x_range, y=ols_intercept + ols_slope*x_range,
                      mode='lines',
                      line=dict(color='green', dash='dot'),
                      name='ols')
        )

    # Update layout
    fig.update_layout(
        title=f"NCA Plot: {y_name} - {x_name}",
        xaxis_title=x_name,
        yaxis_title=y_name,
        template='plotly_white',
        width=800,
        height=600,
        showlegend=True,
        xaxis=dict(
            range=[Xmin-(Xmax-Xmin)*margin_size,
                  Xmax+(Xmax-Xmin)*margin_size]
        ),
        yaxis=dict(
            range=[Ymin-(Ymax-Ymin)*margin_size,
                  Ymax+(Ymax-Ymin)*margin_size]
        )
    )

    return fig

def create_separate_nca_plots(x: np.ndarray, y: np.ndarray,
                            bootstrap_ce_envelopes: list, ce_peers: np.ndarray,
                            bootstrap_cr_slopes: list, bootstrap_cr_intercepts: list,
                            confidence_level: int,
                            x_name: str = "X", y_name: str = "Y",
                            show_ols: bool = True) -> Tuple[go.Figure, go.Figure]:
    """Create separate plots for CE-FDH and CR-FDH with bootstrap results"""
    Xmin, Xmax = np.min(x), np.max(x)
    Ymin, Ymax = np.min(y), np.max(y)
    margin_size = 0.03
    lower_alpha = (1 - confidence_level/100.0) / 2.0
    upper_alpha = 1 - lower_alpha

    # CE-FDH Plot
    ce_fig = go.Figure()

    # Draw scope lines
    for val, line_name in [(Xmin, 'Xmin'), (Xmax, 'Xmax'), (Ymin, 'Ymin'), (Ymax, 'Ymax')]:
        if line_name.startswith('X'):
            ce_fig.add_vline(x=val, line_dash="dash", line_color="gray", opacity=0.5)
        else:
            ce_fig.add_hline(y=val, line_dash="dash", line_color="gray", opacity=0.5)

    # Plot observations (data points)
    ce_fig.add_trace(
        go.Scatter(x=x, y=y,
                  mode='markers',
                  marker=dict(color='blue', symbol='star', size=8),
                  name='obs')
    )

    # Plot peer points with both peer and observation markers
    peer_x = ce_peers[:, 0]
    peer_y = ce_peers[:, 1]
    for px, py in zip(peer_x, peer_y):
        # Add peer marker (red circle)
        ce_fig.add_trace(
            go.Scatter(x=[px], y=[py],
                      mode='markers',
                      marker=dict(color='red', size=10),
                      name='peer',
                      showlegend=False)
        )
        # Add observation marker (blue star) for the same point
        ce_fig.add_trace(
            go.Scatter(x=[px], y=[py],
                      mode='markers',
                      marker=dict(color='blue', symbol='star', size=8),
                      name='obs',
                      showlegend=False)
        )

    # Plot CE-FDH bootstrap results
    x_common = np.linspace(Xmin, Xmax, 100)
    interpolated_envelopes = []

    for envelope in bootstrap_ce_envelopes:
        # Create interpolation function
        f = interp1d(envelope[:, 0], envelope[:, 1],
                   kind='previous', bounds_error=False,
                   fill_value=(envelope[0, 1], envelope[-1, 1]))
        interpolated_y = f(x_common)
        interpolated_envelopes.append(interpolated_y)

    # Calculate mean and CIs
    interpolated_envelopes = np.array(interpolated_envelopes)
    mean_envelope = np.mean(interpolated_envelopes, axis=0)
    lower_bounds = np.percentile(interpolated_envelopes, lower_alpha * 100, axis=0)
    upper_bounds = np.percentile(interpolated_envelopes, upper_alpha * 100, axis=0)

    # Add confidence interval as shaded area
    ce_fig.add_trace(
        go.Scatter(x=x_common, y=upper_bounds,
                  mode='lines',
                  line=dict(width=0),
                  showlegend=False)
    )
    ce_fig.add_trace(
        go.Scatter(x=x_common, y=lower_bounds,
                  mode='lines',
                  line=dict(width=0),
                  fill='tonexty',
                  fillcolor='rgba(255,0,0,0.2)',
                  name=f'{confidence_level}% CI')
    )

    # Plot mean ceiling line
    ce_fig.add_trace(
        go.Scatter(x=x_common, y=mean_envelope,
                  mode='lines',
                  line=dict(color='red', width=2),
                  name='CE-FDH Mean')
    )

    # Plot OLS line if requested
    if show_ols:
        x_range = np.linspace(Xmin, Xmax, 100)
        ols_slope, ols_intercept = np.polyfit(x, y, 1)
        ce_fig.add_trace(
            go.Scatter(x=x_range, y=ols_intercept + ols_slope*x_range,
                      mode='lines',
                      line=dict(color='green', dash='dot'),
                      name='ols')
        )

    ce_fig.update_layout(
        title=f"CE-FDH Plot: {y_name} - {x_name}",
        xaxis_title=x_name,
        yaxis_title=y_name,
        template='plotly_white',
        width=800,
        height=600,
        showlegend=True,
        xaxis=dict(
            range=[Xmin-(Xmax-Xmin)*margin_size,
                  Xmax+(Xmax-Xmin)*margin_size]
        ),
        yaxis=dict(
            range=[Ymin-(Ymax-Ymin)*margin_size,
                  Ymax+(Ymax-Ymin)*margin_size]
        )
    )

    # CR-FDH Plot
    cr_fig = go.Figure()

    # Draw scope lines
    for val, line_name in [(Xmin, 'Xmin'), (Xmax, 'Xmax'), (Ymin, 'Ymin'), (Ymax, 'Ymax')]:
        if line_name.startswith('X'):
            cr_fig.add_vline(x=val, line_dash="dash", line_color="gray", opacity=0.5)
        else:
            cr_fig.add_hline(y=val, line_dash="dash", line_color="gray", opacity=0.5)

    # Plot observations (data points)
    cr_fig.add_trace(
        go.Scatter(x=x, y=y,
                  mode='markers',
                  marker=dict(color='blue', symbol='star', size=8),
                  name='obs')
    )

    # Plot CR-FDH bootstrap results
    x_range = np.linspace(Xmin, Xmax, 100)
    cr_lines = []
    for slope, intercept in zip(bootstrap_cr_slopes, bootstrap_cr_intercepts):
        if not np.isnan(slope) and not np.isnan(intercept):
            cr_lines.append(intercept + slope*x_range)

    if cr_lines:
        cr_lines = np.array(cr_lines)
        lower_cr = np.percentile(cr_lines, lower_alpha * 100, axis=0)
        upper_cr = np.percentile(cr_lines, upper_alpha * 100, axis=0)
        mean_cr_slope = np.nanmean(bootstrap_cr_slopes)
        mean_cr_intercept = np.nanmean(bootstrap_cr_intercepts)
        mean_cr_line = mean_cr_intercept + mean_cr_slope * x_range

        # Add confidence interval
        cr_fig.add_trace(
            go.Scatter(x=x_range, y=upper_cr,
                      mode='lines',
                      line=dict(width=0),
                      showlegend=False)
        )
        cr_fig.add_trace(
            go.Scatter(x=x_range, y=lower_cr,
                      mode='lines',
                      line=dict(width=0),
                      fill='tonexty',
                      fillcolor='rgba(255,165,0,0.2)',
                      name=f'{confidence_level}% CI')
        )

        # Add mean CR-FDH line
        cr_fig.add_trace(
            go.Scatter(x=x_range, y=mean_cr_line,
                      mode='lines',
                      line=dict(color='orange', width=2),
                      name='CR-FDH Mean')
        )

    # Plot OLS line if requested
    if show_ols:
        ols_slope, ols_intercept = np.polyfit(x, y, 1)
        cr_fig.add_trace(
            go.Scatter(x=x_range, y=ols_intercept + ols_slope*x_range,
                      mode='lines',
                      line=dict(color='green', dash='dot'),
                      name='ols')
        )

    cr_fig.update_layout(
        title=f"CR-FDH Plot: {y_name} - {x_name}",
        xaxis_title=x_name,
        yaxis_title=y_name,
        template='plotly_white',
        width=800,
        height=600,
        showlegend=True,
        xaxis=dict(
            range=[Xmin-(Xmax-Xmin)*margin_size,
                  Xmax+(Xmax-Xmin)*margin_size]
        ),
        yaxis=dict(
            range=[Ymin-(Ymax-Ymin)*margin_size,
                  Ymax+(Ymax-Ymin)*margin_size]
        )
    )

    return ce_fig, cr_fig

#############################################
# Additional CE-FDH Inefficiency Computations
#############################################

def compute_ineffs_ce(Xmin, Xmax, Ymin, Ymax, Scope, peers, flip_x=False, flip_y=False):
    # If <2 peers, can't form a polygon
    if peers.shape[0] < 2:
        return np.nan, np.nan, np.nan, np.nan
    # For CE-FDH: x_lim from last peer, y_lim from first peer
    x_lim = peers[-1,0]
    y_lim = peers[0,1]
    return p_ineff(Xmin, Xmax, Ymin, Ymax, Scope, x_lim, y_lim, flip_x, flip_y)

def p_ineff(Xmin, Xmax, Ymin, Ymax, Scope, x_lim, y_lim, flip_x, flip_y):
    # Simplified logic assuming top-left corner (no flip):
    x_lim_clamped = np.clip(x_lim, Xmin, Xmax)
    y_lim_clamped = np.clip(y_lim, Ymin, Ymax)

    ineff_x = (Xmax - x_lim_clamped)/(Xmax - Xmin)
    ineff_y = (y_lim_clamped - Ymin)/(Ymax - Ymin)

    rel_ineff = (ineff_x + ineff_y - ineff_x*ineff_y)*100.0
    abs_ineff = (rel_ineff/100.0)*Scope
    cond_ineff = ineff_x*100.0
    outc_ineff = ineff_y*100.0
    return abs_ineff, rel_ineff, cond_ineff, outc_ineff

def process_data(df, x_col, y_col, transformation=None):
    x = pd.to_numeric(df[x_col], errors='coerce')
    y = pd.to_numeric(df[y_col], errors='coerce')
    mask = ~(x.isna() | y.isna())
    x = x[mask].values
    y = y[mask].values
    if transformation == 'log':
        if np.any(x <= 0) or np.any(y <= 0):
            raise ValueError("Log transform error: Data must be positive.")
        x = np.log(x)
        y = np.log(y)
    elif transformation == 'sqrt':
        if np.any(x < 0) or np.any(y < 0):
            raise ValueError("Sqrt transform error: Data must be non-negative.")
        x = np.sqrt(x)
        y = np.sqrt(y)
    check_data_validity(x, y)
    return x, y

###############################################
# Permutation Test for p-value and p-accuracy #
###############################################

def permutation_test(x, y, method, Xmin, Xmax, Ymin, Ymax, Scope, ce_peers_=None, cr_slope=None, cr_intercept=None,
                     n_iter=200, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    # Compute observed effect size
    if method == 'CE-FDH':
        ce_es = CE_FDH_effect_size(Xmin, Xmax, Ymin, Ymax, ce_peers_)
        observed = ce_es
    elif method == 'CR-FDH':
        cr_es = CR_FDH_effect_size(Xmin, Xmax, Ymin, Ymax, cr_slope, cr_intercept)
        observed = cr_es
    else:
        return None, None

    sim_effects = []
    for _ in range(n_iter):
        perm_y = rng.permutation(y)
        arr = create_sorted_array(x, perm_y)
        ce_env = CE_FDH_envelope_list(arr)
        ce_p = CE_FDH_peers(ce_env)
        if method == 'CE-FDH':
            es = CE_FDH_effect_size(Xmin, Xmax, Ymin, Ymax, ce_p)
        else:
            # method == 'CR-FDH': first CE-FDH then CR
            if ce_p.shape[0] > 1:
                s, i = OLS_params_from_peers(ce_p)
                es = CR_FDH_effect_size(Xmin, Xmax, Ymin, Ymax, s, i)
            else:
                es = np.nan
        sim_effects.append(es)

    sim_effects = np.array(sim_effects)
    sim_effects = sim_effects[~np.isnan(sim_effects)]
    if len(sim_effects) < 10:
        return np.nan, np.nan

    p_val = 1 - np.mean(sim_effects < observed)
    p_val = max(p_val, 1.0/n_iter)
    z=1.96
    p_accuracy = z*np.sqrt(p_val*(1-p_val)/n_iter)
    return p_val, p_accuracy

#########################################
# Compute NCA Metrics with All Additions #
#########################################

def compute_ineff_line(Xmin, Xmax, Ymin, Ymax, slope, intercept, flip_x, flip_y):
    y_for_Xmax = slope * Xmax + intercept
    y_for_Xmin = slope * Xmin + intercept

    if y_for_Xmax > Ymax:
        X_Cmax = (Ymax - intercept)/slope
    else:
        X_Cmax = Xmax
    if y_for_Xmin < Ymin:
        Y_Cmin = Ymin
    else:
        Y_Cmin = y_for_Xmin

    ineff_x = (Xmax - X_Cmax)/(Xmax - Xmin)
    ineff_y = (Y_Cmin - Ymin)/(Ymax - Ymin)
    rel_ineff = (ineff_x + ineff_y - ineff_x*ineff_y)*100.0
    abs_ineff = rel_ineff/100.0 * (Xmax - Xmin)*(Ymax - Ymin)
    cond_ineff = ineff_x*100.0
    outc_ineff = ineff_y*100.0
    return abs_ineff, rel_ineff, cond_ineff, outc_ineff

def compute_nca_metrics(x, y, Xmin, Xmax, Ymin, Ymax, Scope,
                        ceiling, baseline_ceiling, slope, intercept, above, flip_x, flip_y,
                        method=None, peers=None, p_value=None, p_accuracy=None):
    N = len(x)
    effect_size = ceiling / Scope if Scope>0 else np.nan
    c_accuracy = 100 * (N - above) / N if N > 0 else np.nan

    fit = None
    if baseline_ceiling is not None and baseline_ceiling>0:
        if np.isclose(ceiling, baseline_ceiling):
            fit = 100.0
        else:
            fit = 100 - 100 * abs(ceiling - baseline_ceiling)/baseline_ceiling
    else:
        if method == 'CE-FDH':
            fit = 100.0

    if method == 'CE-FDH':
        abs_ineff, rel_ineff, cond_ineff, outc_ineff = compute_ineffs_ce(Xmin, Xmax, Ymin, Ymax, Scope, peers, flip_x, flip_y)
    elif method == 'CR-FDH' and slope is not None and intercept is not None:
        abs_ineff, rel_ineff, cond_ineff, outc_ineff = compute_ineff_line(Xmin, Xmax, Ymin, Ymax, slope, intercept, flip_x, flip_y)
    else:
        abs_ineff, rel_ineff, cond_ineff, outc_ineff = np.nan, np.nan, np.nan, np.nan

    results = {
        "Number of observations": N,
        "Scope": Scope,
        "Xmin": Xmin,
        "Xmax": Xmax,
        "Ymin": Ymin,
        "Ymax": Ymax,
        "Ceiling zone": ceiling,
        "Effect size": effect_size,
        "# above": above,
        "c-accuracy": c_accuracy,
        "Fit": fit,
        "p-value": p_value,
        "p-accuracy": p_accuracy,
        "Slope": slope,
        "Intercept": intercept,
        "Abs. ineff.": abs_ineff,
        "Rel. ineff.": rel_ineff,
        "Condition ineff.": cond_ineff,
        "Outcome ineff.": outc_ineff
    }
    return results

########################################
# Main Streamlit App Logic Starts Here #
########################################

def create_distribution_plots(df, x_col, y_col):
    fig = make_subplots(rows=2, cols=2, subplot_titles=[f"Dist of {x_col}", f"Dist of {y_col}", f"{x_col} vs {y_col}", "Correlation Heatmap"], vertical_spacing=0.2, horizontal_spacing=0.1)
    fig.add_trace(go.Histogram(x=df[x_col], nbinsx=30, marker_color='blue', name=x_col), row=1, col=1)
    fig.update_xaxes(title_text=x_col, row=1, col=1)
    fig.update_yaxes(title_text='Count', row=1, col=1)
    fig.add_trace(go.Histogram(x=df[y_col], nbinsx=30, marker_color='red', name=y_col), row=1, col=2)
    fig.update_xaxes(title_text=y_col, row=1, col=2)
    fig.update_yaxes(title_text='Count', row=1, col=2)
    fig.add_trace(go.Scatter(x=df[x_col], y=df[y_col], mode='markers', marker_color='purple'), row=2, col=1)
    fig.update_xaxes(title_text=x_col, row=2, col=1)
    fig.update_yaxes(title_text=y_col, row=2, col=1)
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()
    fig.add_trace(go.Heatmap(z=corr.values, x=corr.columns, y=corr.columns, colorscale='RdBu', reversescale=True), row=2, col=2)
    fig.update_layout(title="Data Distributions and Correlation", template='plotly_white', height=800, width=900)
    return fig

def create_residual_plots(x, residuals):
    fig = make_subplots(rows=1, cols=2, subplot_titles=["Residuals vs X (with LOESS)", "Residual Distribution"], column_widths=[0.6, 0.4])
    fig.add_trace(go.Scatter(x=x, y=residuals, mode='markers', marker=dict(color='blue', size=5, opacity=0.7)), row=1, col=1)
    smoothed = sm.nonparametric.lowess(residuals, x, frac=0.3)
    fig.add_trace(go.Scatter(x=smoothed[:, 0], y=smoothed[:, 1], mode='lines', line=dict(color='red')), row=1, col=1)
    fig.update_xaxes(title_text='X', row=1, col=1)
    fig.update_yaxes(title_text='Residuals', row=1, col=1)
    fig.add_trace(go.Histogram(x=residuals, nbinsx=30, marker_color='blue', opacity=0.7), row=1, col=2)
    fig.update_xaxes(title_text='Residual Value', row=1, col=2)
    fig.update_yaxes(title_text='Count', row=1, col=2)
    fig.update_layout(title="Residual Analysis", template='plotly_white', width=900, height=400)
    return fig

def create_qq_plot(residuals):
    sorted_res = np.sort(residuals)
    n = len(sorted_res)
    theoretical = stats.norm.ppf((np.arange(n) + 0.5) / n)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=theoretical, y=sorted_res, mode='markers', marker=dict(color='blue', size=5, opacity=0.7)))
    min_val = min(theoretical.min(), sorted_res.min())
    max_val = max(theoretical.max(), sorted_res.max())
    fig.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val], mode='lines', line=dict(color='red', dash='dash')))
    fig.update_layout(title="QQ Plot of Residuals", xaxis_title="Theoretical Quantiles", yaxis_title="Sample Quantiles", template='plotly_white', width=400, height=400)
    return fig

def create_bottleneck_plot(df_bottleneck, x_name, y_name, confidence_level=None):
    """Create an interactive plot visualizing the bottleneck table"""
    fig = go.Figure()
    
    # Plot mean/ceiling line
    if confidence_level is not None:
        # With bootstrap results
        fig.add_trace(
            go.Scatter(
                x=df_bottleneck["X Value"],
                y=df_bottleneck["Mean Ceiling Y"],
                mode='lines',
                name='Mean Ceiling',
                line=dict(color='red', width=2)
            )
        )
        
        # Add confidence intervals
        fig.add_trace(
            go.Scatter(
                x=df_bottleneck["X Value"],
                y=df_bottleneck[f"CI Upper ({confidence_level}%)"],
                mode='lines',
                line=dict(width=0),
                showlegend=False
            )
        )
        fig.add_trace(
            go.Scatter(
                x=df_bottleneck["X Value"],
                y=df_bottleneck[f"CI Lower ({confidence_level}%)"],
                mode='lines',
                line=dict(width=0),
                fill='tonexty',
                fillcolor='rgba(255,0,0,0.2)',
                name=f'{confidence_level}% CI'
            )
        )
    else:
        # Without bootstrap
        fig.add_trace(
            go.Scatter(
                x=df_bottleneck["X Value"],
                y=df_bottleneck["Ceiling Y"],
                mode='lines',
                name='Ceiling',
                line=dict(color='red', width=2)
            )
        )
    
    fig.update_layout(
        title=f"Bottleneck Analysis: Required {y_name} for given {x_name}",
        xaxis_title=x_name,
        yaxis_title=f"Required {y_name}",
        template='plotly_white',
        width=800,
        height=500
    )
    
    return fig

def main():
    page = st.sidebar.selectbox("Select Page", ["Meta-Analysis Data Generation", "NCA Analysis"])
    if page == "Meta-Analysis Data Generation":
        st.title("Generate Simulated Data from Meta-Analysis")
        st.markdown("""
        This page allows you to generate simulated meta-analysis data with specified parameters.
        The data generation process uses multivariate normal distributions with optional correlation structure.
        Note that skewness and kurtosis inputs are approximate due to the use of normal distributions with clipping.
        """)
        
        n_moderators = st.number_input("Number of Moderator Levels", min_value=1, value=2,
                                     help="Number of distinct moderator categories or groups in your meta-analysis")
        
        moderator_levels = [st.text_input(f"Moderator Level {i + 1}", value=f"Level {i + 1}",
                                         help=f"Name or label for moderator level {i + 1}") 
                           for i in range(n_moderators)]
        
        mean_x = [st.number_input(f"Mean X ({mod})", value=4.0,
                                help=f"Mean value of X variable for {mod}") 
                 for mod in moderator_levels]
        
        sd_x = [st.number_input(f"SD X ({mod})", value=1.0,
                              help=f"Standard deviation of X variable for {mod}")
                for mod in moderator_levels]
        
        min_x_ = [st.number_input(f"Min X ({mod})", value=1.0,
                                help=f"Minimum allowed value of X for {mod}")
                  for mod in moderator_levels]
        
        max_x_ = [st.number_input(f"Max X ({mod})", value=7.0,
                                help=f"Maximum allowed value of X for {mod}")
                  for mod in moderator_levels]
        
        mean_y = [st.number_input(f"Mean Effect Size (Y) ({mod})", value=4.0) for mod in moderator_levels]
        sd_y = [st.number_input(f"SD Effect Size (Y) ({mod})", value=1.0) for mod in moderator_levels]
        min_y_ = [st.number_input(f"Min Effect Size (Y) ({mod})", value=1.0) for mod in moderator_levels]
        max_y_ = [st.number_input(f"Max Effect Size (Y) ({mod})", value=7.0) for mod in moderator_levels]
        n_studies = [st.number_input(f"Number of Studies ({mod})", min_value=1, value=100) for mod in moderator_levels]
        corr_xy_option = st.checkbox("Specify X-Y Correlation (per moderator)", value=True)
        if corr_xy_option:
            corr_xy = [st.number_input(f"Correlation X-Y ({mod})", value=0.5, min_value=-1.0, max_value=1.0) for mod in moderator_levels]
        else:
            corr_xy = None
        skew_kurt_option = st.checkbox("Specify Skewness and Kurtosis (Optional)", value=True)
        if skew_kurt_option:
            skew_x_ = [st.number_input(f"Skewness X ({mod})", value=0.5) for mod in moderator_levels]
            kurt_x_ = [st.number_input(f"Kurtosis X ({mod})", value=3.0) for mod in moderator_levels]
            skew_y_ = [st.number_input(f"Skewness Y ({mod})", value=-0.5) for mod in moderator_levels]
            kurt_y_ = [st.number_input(f"Kurtosis Y ({mod})", value=2.5) for mod in moderator_levels]
        else:
            skew_x_, kurt_x_, skew_y_, kurt_y_ = None, None, None, None

        if st.button("Generate Data"):
            simulated_data = []
            np.random.seed(0)
            for i in range(len(moderator_levels)):
                if corr_xy is not None:
                    mean = [mean_x[i], mean_y[i]]
                    cov = [[sd_x[i]**2, corr_xy[i]*sd_x[i]*sd_y[i]],
                           [corr_xy[i]*sd_x[i]*sd_y[i], sd_y[i]**2]]
                    Xtemp, Ytemp = np.random.multivariate_normal(mean, cov, n_studies[i]).T
                else:
                    Xtemp = np.random.normal(mean_x[i], sd_x[i], n_studies[i])
                    Ytemp = np.random.normal(mean_y[i], sd_y[i], n_studies[i])
                Xtemp = np.clip(Xtemp, min_x_[i], max_x_[i])
                Ytemp = np.clip(Ytemp, min_y_[i], max_y_[i])
                df_temp = pd.DataFrame({"Moderator": moderator_levels[i], "X": Xtemp, "Y": Ytemp})
                simulated_data.append(df_temp)
            simulated_df = pd.concat(simulated_data, ignore_index=True)
            st.write("Simulated Data:")
            st.dataframe(simulated_df)
            fig = go.Figure()
            for mod in moderator_levels:
                mod_df = simulated_df[simulated_df['Moderator'] == mod]
                fig.add_trace(go.Scatter(x=mod_df['X'], y=mod_df['Y'], mode='markers', name=mod))
            fig.update_layout(xaxis_title="X", yaxis_title="Effect Size (Y)", title="Scatter Plot of Simulated Data")
            st.plotly_chart(fig)
            csv_bytes = simulated_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Simulated Data", data=csv_bytes, file_name='simulated_meta_data.csv', mime='text/csv')

    elif page == "NCA Analysis":
        st.title("Advanced NCA Analysis Tool")
        st.markdown("""
        This tool performs Necessary Condition Analysis (NCA) with advanced features including:
        - Bootstrap analysis for confidence intervals
        - Permutation tests for significance
        - Both CE-FDH and CR-FDH methods
        - Comprehensive residual analysis
        - Bottleneck analysis
        
        Upload your data in CSV format with at least two numeric columns.
        """)
        
        uploaded_file = st.file_uploader("Upload your CSV", type=["csv"],
                                       help="Your data file should be in CSV format with at least two numeric columns")
        
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.write("Data Preview:")
            st.dataframe(df.head())
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) < 2:
                st.error("Need ≥2 numeric cols.")
            else:
                x_col = st.selectbox("Select X Variable", numeric_cols,
                                   help="Independent variable (condition)")
                y_col = st.selectbox("Select Y Variable", 
                                   [c for c in numeric_cols if c != x_col],
                                   help="Dependent variable (outcome)")
                transformation = st.selectbox("Apply Data Transformation (Optional)",
                                           [None, 'log', 'sqrt'],
                                           help="Transform data before analysis. Use log for multiplicative relationships, sqrt for moderate skewness")
                fig_dist = create_distribution_plots(df, x_col, y_col)
                st.write("### Data Exploration")
                st.plotly_chart(fig_dist, use_container_width=True)

                # Bootstrap parameters
                st.sidebar.subheader("Analysis Settings")
                analysis_type = st.sidebar.radio(
                    "Analysis Type",
                    ["Standard NCA (No Bootstrap)", "NCA with Bootstrap"],
                    index=1  # Default to bootstrap
                )
                
                if analysis_type == "NCA with Bootstrap":
                    st.sidebar.subheader("Bootstrap Settings")
                    bootstrap_iterations = st.sidebar.slider("Bootstrap Iterations", 100, 10000, 1000)  # Default to 1000
                    confidence_level = st.sidebar.slider("Confidence Level (%)", 80, 99, 95)  # Default to 95%
                
                # Permutation test parameters
                test_rep = st.number_input("Permutation Test Repetitions (for p-values)", min_value=0, value=1000)
                
                run_analysis = st.button("Run NCA Analysis")

                if run_analysis:
                    try:
                        x, y = process_data(df, x_col, y_col, transformation)
                        Xmin, Xmax, Ymin, Ymax, Scope = define_scope_limits(x, y)
                        arr = create_sorted_array(x, y)

                        # Compute initial CE-FDH envelope and peers
                        ce_envelope = CE_FDH_envelope_list(arr)
                        ce_peers_ = CE_FDH_peers(ce_envelope)

                        if analysis_type == "NCA with Bootstrap":
                            # Initialize bootstrap storage
                            bootstrap_ce_es = []
                            bootstrap_cr_es = []
                            bootstrap_ce_envelopes = []
                            bootstrap_cr_slopes = []
                            bootstrap_cr_intercepts = []

                            # Progress bar for bootstrap
                            my_bar = st.progress(0)
                            st.write("Running bootstrap analysis...")

                            # Bootstrap loop
                            for i in range(bootstrap_iterations):
                                # Resample data
                                indices = np.random.choice(len(x), size=len(x), replace=True)
                                x_resampled = x[indices]
                                y_resampled = y[indices]

                                # Compute bootstrap metrics
                                arr_b = create_sorted_array(x_resampled, y_resampled)
                                ce_envelope_b = CE_FDH_envelope_list(arr_b)
                                ce_peers_b = CE_FDH_peers(ce_envelope_b)
                                ce_es_b = CE_FDH_effect_size(Xmin, Xmax, Ymin, Ymax, ce_peers_b)

                                # Always append CE-FDH results
                                bootstrap_ce_es.append(ce_es_b)
                                bootstrap_ce_envelopes.append(ce_envelope_b)

                                # Compute CR-FDH metrics, append NaN if not computable
                                if ce_peers_b.shape[0] > 1:
                                    try:
                                        cr_slope_b, cr_intercept_b = OLS_params_from_peers(ce_peers_b)
                                        cr_es_b = CR_FDH_effect_size(Xmin, Xmax, Ymin, Ymax, cr_slope_b, cr_intercept_b)
                                    except Exception:
                                        cr_slope_b, cr_intercept_b, cr_es_b = np.nan, np.nan, np.nan
                                else:
                                    cr_slope_b, cr_intercept_b, cr_es_b = np.nan, np.nan, np.nan
                                
                                bootstrap_cr_es.append(cr_es_b)
                                bootstrap_cr_slopes.append(cr_slope_b)
                                bootstrap_cr_intercepts.append(cr_intercept_b)

                                my_bar.progress((i + 1) / bootstrap_iterations)

                            # Calculate confidence intervals
                            lower_alpha = (1 - confidence_level/100.0) / 2.0
                            upper_alpha = 1 - lower_alpha

                            def get_bootstrap_ci(data):
                                data = [d for d in data if not np.isnan(d)]
                                if len(data) < 2:
                                    return np.nan, np.nan
                                return np.percentile(data, [lower_alpha * 100, upper_alpha * 100])

                            ce_es_ci = get_bootstrap_ci(bootstrap_ce_es)
                            cr_es_ci = get_bootstrap_ci(bootstrap_cr_es)
                            cr_slope_ci = get_bootstrap_ci(bootstrap_cr_slopes)
                            cr_intercept_ci = get_bootstrap_ci(bootstrap_cr_intercepts)

                            # Calculate mean values from bootstrap
                            mean_ce_es = np.mean(bootstrap_ce_es)
                            mean_cr_es = np.nanmean(bootstrap_cr_es)
                            mean_cr_slope = np.nanmean(bootstrap_cr_slopes)
                            mean_cr_intercept = np.nanmean(bootstrap_cr_intercepts)

                            # Use bootstrap means for ceilings
                            ce_ceiling = mean_ce_es * Scope
                            cr_ceiling = mean_cr_es * Scope if not np.isnan(mean_cr_es) else np.nan

                            # Compute CE-FDH line
                            if ce_peers_.shape[0] > 1:
                                cr_above = np.sum(y > (mean_cr_intercept + mean_cr_slope*x))
                            else:
                                cr_above = np.nan

                            ce_above = 0  # By definition CE-FDH line is top envelope, no above

                        else:  # Standard NCA without bootstrap
                            st.write("Running standard NCA analysis without bootstrapping...")

                            # Calculate CE-FDH metrics directly
                            ce_es = CE_FDH_effect_size(Xmin, Xmax, Ymin, Ymax, ce_peers_)
                            ce_ceiling = ce_es * Scope
                            ce_above = 0  # By definition

                            # Calculate CR-FDH metrics directly
                            if ce_peers_.shape[0] > 1:
                                cr_slope, cr_intercept = OLS_params_from_peers(ce_peers_)
                                cr_es = CR_FDH_effect_size(Xmin, Xmax, Ymin, Ymax, cr_slope, cr_intercept)
                                cr_ceiling = cr_es * Scope
                                cr_above = np.sum(y > (cr_intercept + cr_slope*x))
                                mean_cr_slope = cr_slope
                                mean_cr_intercept = cr_intercept
                            else:
                                cr_slope = cr_intercept = np.nan
                                cr_es = cr_ceiling = cr_above = np.nan
                                mean_cr_slope = mean_cr_intercept = np.nan

                            # Set bootstrap-related variables to None for non-bootstrap analysis
                            bootstrap_ce_envelopes = None
                            bootstrap_cr_slopes = None
                            bootstrap_cr_intercepts = None
                            ce_es_ci = (np.nan, np.nan)
                            cr_es_ci = (np.nan, np.nan)
                            cr_slope_ci = (np.nan, np.nan)
                            cr_intercept_ci = (np.nan, np.nan)
                            mean_ce_es = ce_es
                            mean_cr_es = cr_es

                        # Permutation tests (run for both analysis types):
                        rng = np.random.default_rng(0)
                        p_val_ce, p_acc_ce = permutation_test(x, y, 'CE-FDH', Xmin, Xmax, Ymin, Ymax, Scope,
                                                              ce_peers_=ce_peers_, n_iter=test_rep, rng=rng)
                        p_val_cr, p_acc_cr = np.nan, np.nan
                        if not np.isnan(mean_cr_slope):
                            p_val_cr, p_acc_cr = permutation_test(x, y, 'CR-FDH', Xmin, Xmax, Ymin, Ymax, Scope,
                                                                  ce_peers_=ce_peers_, cr_slope=mean_cr_slope,
                                                                  cr_intercept=mean_cr_intercept,
                                                                  n_iter=test_rep, rng=rng)

                        # Compute metrics:
                        ce_metrics = compute_nca_metrics(x, y, Xmin, Xmax, Ymin, Ymax, Scope,
                                                         ceiling=ce_ceiling, baseline_ceiling=None,
                                                         slope=None, intercept=None,
                                                         above=ce_above, flip_x=False, flip_y=False,
                                                         method='CE-FDH', peers=ce_peers_,
                                                         p_value=p_val_ce, p_accuracy=p_acc_ce)

                        cr_metrics = compute_nca_metrics(x, y, Xmin, Xmax, Ymin, Ymax, Scope,
                                                         ceiling=cr_ceiling, baseline_ceiling=ce_ceiling,
                                                         slope=mean_cr_slope, intercept=mean_cr_intercept,
                                                         above=cr_above if not np.isnan(cr_above) else 0,
                                                         flip_x=False, flip_y=False,
                                                         method='CR-FDH', peers=None,
                                                         p_value=p_val_cr, p_accuracy=p_acc_cr)

                        st.write("#### NCA Plot")
                        if analysis_type == "NCA with Bootstrap":
                            ce_fig, cr_fig = create_separate_nca_plots(
                                x=x, y=y,
                                bootstrap_ce_envelopes=bootstrap_ce_envelopes,
                                ce_peers=ce_peers_,
                                bootstrap_cr_slopes=bootstrap_cr_slopes,
                                bootstrap_cr_intercepts=bootstrap_cr_intercepts,
                                confidence_level=confidence_level,
                                x_name=x_col,
                                y_name=y_col,
                                show_ols=True
                            )
                            st.plotly_chart(ce_fig)
                            st.plotly_chart(cr_fig)

                        else:  # Standard NCA
                            fig = create_nca_plot(
                                x=x, y=y,
                                ce_envelope=ce_envelope,
                                ce_peers=ce_peers_,
                                cr_slope=mean_cr_slope,
                                cr_intercept=mean_cr_intercept,
                                x_name=x_col,
                                y_name=y_col,
                                show_ols=True
                            )
                            st.plotly_chart(fig)

                        # Remove the old matplotlib plotting code here
                        # st.pyplot(fig)

                        # Add diagnostic plot showing all bootstrap ceilings (only for bootstrap analysis)
                        if analysis_type == "NCA with Bootstrap":
                            st.write("#### Diagnostic Plot: All Bootstrap Ceilings")
                            fig_diag = go.Figure()

                            # Plot original data points
                            fig_diag.add_trace(go.Scatter(
                                x=x, y=y,
                                mode='markers',
                                name='Data Points',
                                marker=dict(color='black', size=5)
                            ))

                            # Plot all CE-FDH envelopes with low opacity
                            for envelope in bootstrap_ce_envelopes:
                                fig_diag.add_trace(go.Scatter(
                                    x=envelope[:, 0],
                                    y=envelope[:, 1],
                                    mode='lines',
                                    line=dict(color='blue', width=1),
                                    opacity=0.1,
                                    showlegend=False
                                ))

                            # Plot all CR-FDH lines with low opacity
                            x_range = np.linspace(Xmin, Xmax, 100)
                            for slope, intercept in zip(bootstrap_cr_slopes, bootstrap_cr_intercepts):
                                if not np.isnan(slope) and not np.isnan(intercept):
                                    fig_diag.add_trace(go.Scatter(
                                        x=x_range,
                                        y=slope * x_range + intercept,
                                        mode='lines',
                                        line=dict(color='red', width=1),
                                        opacity=0.1,
                                        showlegend=False
                                    ))

                            # Add mean lines with full opacity
                            # For CE-FDH, interpolate each envelope to common x points
                            x_common = np.linspace(Xmin, Xmax, 100)
                            interpolated_envelopes = []

                            for envelope in bootstrap_ce_envelopes:
                                # Create interpolation function
                                f = interp1d(envelope[:, 0], envelope[:, 1],
                                           kind='previous', bounds_error=False,
                                           fill_value=(envelope[0, 1], envelope[-1, 1]))
                                interpolated_y = f(x_common)
                                interpolated_envelopes.append(interpolated_y)

                            # Calculate mean envelope
                            mean_envelope_y = np.mean(interpolated_envelopes, axis=0)

                            # Plot mean CE-FDH
                            fig_diag.add_trace(go.Scatter(
                                x=x_common,
                                y=mean_envelope_y,
                                mode='lines',
                                line=dict(color='blue', width=2),
                                name='Mean CE-FDH'
                            ))

                            # Mean CR-FDH line
                            mean_cr_y = mean_cr_slope * x_range + mean_cr_intercept
                            fig_diag.add_trace(go.Scatter(
                                x=x_range,
                                y=mean_cr_y,
                                mode='lines',
                                line=dict(color='red', width=2),
                                name='Mean CR-FDH'
                            ))

                            fig_diag.update_layout(
                                title='All Bootstrap Ceilings',
                                xaxis_title='X',
                                yaxis_title='Y',
                                showlegend=True
                            )
                            st.plotly_chart(fig_diag)

                        # Summarize results in a DataFrame:
                        data_rows = []
                        for method_name, mets, es_ci, slope_ci, intercept_ci in zip(
                            ["CE-FDH", "CR-FDH"],
                            [ce_metrics, cr_metrics],
                            [ce_es_ci, cr_es_ci],
                            [None, cr_slope_ci],
                            [None, cr_intercept_ci]
                        ):
                            if analysis_type == "NCA with Bootstrap":
                                row_data = {
                                    "Method": method_name,
                                    "Effect Size": f"{mets['Effect size']:.3f} ({es_ci[0]:.3f}, {es_ci[1]:.3f})" if not np.isnan(mets['Effect size']) and not np.any(np.isnan(es_ci)) else "",
                                    "Ceiling zone": f"{mets['Ceiling zone']:.3f}" if not np.isnan(mets['Ceiling zone']) else "",
                                    "# above": mets['# above'],
                                    "c-accuracy (%)": f"{mets['c-accuracy']:.1f}" if not np.isnan(mets['c-accuracy']) else "",
                                    "Fit (%)": f"{mets['Fit']:.1f}" if mets['Fit'] is not None and not np.isnan(mets['Fit']) else "",
                                    "Slope": f"{mets['Slope']:.3f} ({slope_ci[0]:.3f}, {slope_ci[1]:.3f})" if slope_ci is not None and not np.any(np.isnan(slope_ci)) and mets['Slope'] is not None and not np.isnan(mets['Slope']) else "",
                                    "Intercept": f"{mets['Intercept']:.3f} ({intercept_ci[0]:.3f}, {intercept_ci[1]:.3f})" if intercept_ci is not None and not np.any(np.isnan(intercept_ci)) and mets['Intercept'] is not None and not np.isnan(mets['Intercept']) else "",
                                    "Abs. ineff.": f"{mets['Abs. ineff.']:.3f}" if not np.isnan(mets['Abs. ineff.']) else "",
                                    "Rel. ineff.": f"{mets['Rel. ineff.']:.3f}" if not np.isnan(mets['Rel. ineff.']) else "",
                                    "Condition ineff.": f"{mets['Condition ineff.']:.3f}" if not np.isnan(mets['Condition ineff.']) else "",
                                    "Outcome ineff.": f"{mets['Outcome ineff.']:.3f}" if not np.isnan(mets['Outcome ineff.']) else "",
                                    "p-value": f"{mets['p-value']:.3f}" if mets['p-value'] is not None and not np.isnan(mets['p-value']) else "",
                                    "p-accuracy": f"{mets['p-accuracy']:.3f}" if mets['p-accuracy'] is not None and not np.isnan(mets['p-accuracy']) else ""
                                }
                            else:
                                row_data = {
                                    "Method": method_name,
                                    "Effect Size": f"{mets['Effect size']:.3f}" if not np.isnan(mets['Effect size']) else "",
                                    "Ceiling zone": f"{mets['Ceiling zone']:.3f}" if not np.isnan(mets['Ceiling zone']) else "",
                                    "# above": mets['# above'],
                                    "c-accuracy (%)": f"{mets['c-accuracy']:.1f}" if not np.isnan(mets['c-accuracy']) else "",
                                    "Fit (%)": f"{mets['Fit']:.1f}" if mets['Fit'] is not None and not np.isnan(mets['Fit']) else "",
                                    "Slope": f"{mets['Slope']:.3f}" if mets['Slope'] is not None and not np.isnan(mets['Slope']) else "",
                                    "Intercept": f"{mets['Intercept']:.3f}" if mets['Intercept'] is not None and not np.isnan(mets['Intercept']) else "",
                                    "Abs. ineff.": f"{mets['Abs. ineff.']:.3f}" if not np.isnan(mets['Abs. ineff.']) else "",
                                    "Rel. ineff.": f"{mets['Rel. ineff.']:.3f}" if not np.isnan(mets['Rel. ineff.']) else "",
                                    "Condition ineff.": f"{mets['Condition ineff.']:.3f}" if not np.isnan(mets['Condition ineff.']) else "",
                                    "Outcome ineff.": f"{mets['Outcome ineff.']:.3f}" if not np.isnan(mets['Outcome ineff.']) else "",
                                    "p-value": f"{mets['p-value']:.3f}" if mets['p-value'] is not None and not np.isnan(mets['p-value']) else "",
                                    "p-accuracy": f"{mets['p-accuracy']:.3f}" if mets['p-accuracy'] is not None and not np.isnan(mets['p-accuracy']) else ""
                                }
                            data_rows.append(row_data)
                        results_df = pd.DataFrame(data_rows)
                        st.write("### Results Summary")
                        st.dataframe(results_df)

                        # Create bottleneck table
                        st.write("### Bottleneck Analysis")

                        if analysis_type == "NCA with Bootstrap":
                            # CE-FDH Bottleneck Data with bootstrap
                            bottleneck_data_ce = []
                            unique_x_original = sorted(np.unique(x))

                            def get_ce_fdh_y(x_val, ce_envelope):
                                env_x = ce_envelope[:, 0]
                                env_y = ce_envelope[:, 1]
                                if x_val < env_x[0]:
                                    return None
                                for i in range(len(env_x) - 1):
                                    if env_x[i] <= x_val <= env_x[i + 1]:
                                        return env_y[i]
                                if x_val >= env_x[-1]:
                                    return env_y[-1]
                                return None

                            for current_x in unique_x_original:
                                ceiling_ys_at_x = []
                                for b_result in bootstrap_ce_envelopes:
                                    y_val = get_ce_fdh_y(current_x, b_result)
                                    if y_val is not None:
                                        ceiling_ys_at_x.append(y_val)

                                if ceiling_ys_at_x:
                                    mean_y = np.mean(ceiling_ys_at_x)
                                    ci_y = np.percentile(ceiling_ys_at_x, [lower_alpha * 100, upper_alpha * 100])
                                    bottleneck_data_ce.append({
                                        "X Value": current_x,
                                        "Mean Ceiling Y": mean_y,
                                        f"CI Lower ({confidence_level}%)": ci_y[0],
                                        f"CI Upper ({confidence_level}%)": ci_y[1],
                                    })
                        else:
                            # CE-FDH Bottleneck Data without bootstrap
                            bottleneck_data_ce = []
                            unique_x_original = sorted(np.unique(x))

                            def get_ce_fdh_y(x_val, ce_envelope):
                                env_x = ce_envelope[:, 0]
                                env_y = ce_envelope[:, 1]
                                if x_val < env_x[0]:
                                    return None
                                for i in range(len(env_x) - 1):
                                    if env_x[i] <= x_val <= env_x[i + 1]:
                                        return env_y[i]
                                if x_val >= env_x[-1]:
                                    return env_y[-1]
                                return None

                            for current_x in unique_x_original:
                                y_val = get_ce_fdh_y(current_x, ce_envelope)
                                if y_val is not None:
                                    bottleneck_data_ce.append({
                                        "X Value": current_x,
                                        "Ceiling Y": y_val
                                    })

                        df_bottleneck_ce = pd.DataFrame(bottleneck_data_ce)
                        st.write("#### CE-FDH Bottleneck Table")
                        st.dataframe(df_bottleneck_ce)

                        # Add bottleneck visualization
                        st.write("#### CE-FDH Bottleneck Visualization")
                        fig_bottleneck = create_bottleneck_plot(
                            df_bottleneck_ce,
                            x_name=x_col,
                            y_name=y_col,
                            confidence_level=confidence_level if analysis_type == "NCA with Bootstrap" else None
                        )
                        st.plotly_chart(fig_bottleneck)

                        # CR-FDH Bottleneck Information
                        st.write("#### CR-FDH Parameters")
                        if analysis_type == "NCA with Bootstrap":
                            st.write(f"Slope: {mean_cr_slope:.3f} ({cr_slope_ci[0]:.3f}, {cr_slope_ci[1]:.3f})")
                            st.write(f"Intercept: {mean_cr_intercept:.3f} ({cr_intercept_ci[0]:.3f}, {cr_intercept_ci[1]:.3f})")
                        else:
                            st.write(f"Slope: {mean_cr_slope:.3f}")
                            st.write(f"Intercept: {mean_cr_intercept:.3f}")

                        # Residual analysis
                        def ce_fdh_ceiling_value(x_val):
                            env_x = ce_envelope[:,0]
                            env_y = ce_envelope[:,1]
                            if x_val <= env_x[0]:
                                return env_y[0]
                            for i in range(1, len(env_x)):
                                if env_x[i] >= x_val:
                                    return env_y[i-1]
                            return env_y[-1]

                        ce_ceiling_vals = np.array([ce_fdh_ceiling_value(xv) for xv in x])
                        if not np.isnan(mean_cr_slope):
                            cr_ceiling_vals = mean_cr_intercept + mean_cr_slope*x
                        else:
                            cr_ceiling_vals = np.full_like(x, np.nan)

                        residuals_ce = y - ce_ceiling_vals
                        residuals_cr = y - cr_ceiling_vals

                        st.write("### Residual Analysis")
                        for method_name, residuals in zip(["CE-FDH","CR-FDH"], [residuals_ce, residuals_cr]):
                            st.write(f"#### {method_name}")
                            fig_res = create_residual_plots(x, residuals)
                            st.plotly_chart(fig_res, use_container_width=True)
                            fig_qq = create_qq_plot(residuals)
                            st.plotly_chart(fig_qq, use_container_width=True)
                            if len(residuals) < 5000:
                                _, p_val_shapiro = stats.shapiro(residuals)
                                st.write(f"Shapiro-Wilk p={p_val_shapiro:.3f}")
                                if p_val_shapiro < 0.05:
                                    st.write("Residuals deviate significantly from normality.")
                                else:
                                    st.write("No significant deviation from normality.")
                            else:
                                st.write("No Shapiro-Wilk test due to large N.")

                    except ValueError as e:
                        st.error(f"Error: {e}")

if __name__ == "__main__":
    main()
