"""Agent configuration. Run with python -m figures.agent_configuration."""
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from .style import apply_medagents_style, get_figure_1_colors, get_manchester_colors, plot_pdf, save_figure


def plot_agent_scaling_analysis(ax, df, colors, panel_label='a'):
    """Plot how performance changes with number of agents"""

    # Configuration
    AGENT_MAPPING = {1: 0, 2: 1, 3: 2, 5: 3}

    # Parse experiment names
    def parse_exp_name(exp_name):
        parts = exp_name.split('_')
        n_agents = int(parts[0])
        n_rounds = int(parts[2])
        has_search = 'with_search' in exp_name
        return n_agents, n_rounds, has_search

    df[['n_agents', 'n_rounds', 'has_search']] = df['exp_name'].apply(
        lambda x: pd.Series(parse_exp_name(x))
    )

    # Calculate summary statistics
    summary_df = df.groupby(['n_agents', 'n_rounds', 'has_search']).agg({
        'accuracy': ['mean', 'std'],
        'avg_time': ['mean', 'std'],
        'avg_cost': 'mean'
    }).round(2)

    summary_df.columns = ['accuracy_mean', 'accuracy_std', 'time_mean', 'time_std', 'cost_mean']
    summary_df = summary_df.reset_index()
    summary_df['n_agents_mapped'] = summary_df['n_agents'].map(AGENT_MAPPING)

    # Plot with search (1 round)
    agent_data_search = summary_df[(summary_df['n_rounds'] == 1) & (summary_df['has_search'] == True)]
    if not agent_data_search.empty:
        ax.plot(agent_data_search['n_agents_mapped'], agent_data_search['accuracy_mean'],
               marker='o', linewidth=3, markersize=10,
               color=colors['metrics']['accuracy'], alpha=0.8,
               markeredgecolor='black', markeredgewidth=2, label='With search')

        ax.fill_between(agent_data_search['n_agents_mapped'],
                       agent_data_search['accuracy_mean'] - agent_data_search['accuracy_std'],
                       agent_data_search['accuracy_mean'] + agent_data_search['accuracy_std'],
                       color=colors['metrics']['accuracy'], alpha=0.2)

        # Add value annotations
        for i, row in agent_data_search.iterrows():
            ax.annotate(f'{row["accuracy_mean"]:.1f}%',
                       (row['n_agents_mapped'], row['accuracy_mean']),
                       textcoords="offset points", xytext=(0,15), ha='center',
                       fontsize=12, fontweight='bold', color='black')

    # Plot without search (1 round)
    agent_data_no_search = summary_df[(summary_df['n_rounds'] == 1) & (summary_df['has_search'] == False)]
    if not agent_data_no_search.empty:
        ax.plot(agent_data_no_search['n_agents_mapped'], agent_data_no_search['accuracy_mean'],
               marker='s', linewidth=3, markersize=10,
               color=colors['metrics']['time'], alpha=0.8,
               markeredgecolor='black', markeredgewidth=2, label='No search')

        ax.fill_between(agent_data_no_search['n_agents_mapped'],
                       agent_data_no_search['accuracy_mean'] - agent_data_no_search['accuracy_std'],
                       agent_data_no_search['accuracy_mean'] + agent_data_no_search['accuracy_std'],
                       color=colors['metrics']['time'], alpha=0.2)

        # Add value annotations
        for i, row in agent_data_no_search.iterrows():
            ax.annotate(f'{row["accuracy_mean"]:.1f}%',
                       (row['n_agents_mapped'], row['accuracy_mean']),
                       textcoords="offset points", xytext=(0,-20), ha='center',
                       fontsize=12, fontweight='bold', color='black')

    # Highlight best performing configuration
    if not summary_df[(summary_df['n_rounds'] == 1)].empty:
        best_accuracy_idx = summary_df[(summary_df['n_rounds'] == 1)]['accuracy_mean'].idxmax()
        best_point = summary_df.loc[best_accuracy_idx]
        best_x = AGENT_MAPPING[best_point['n_agents']]
        best_y = best_point['accuracy_mean']
        ax.plot(best_x, best_y, marker='*', markersize=15, color='gold',
               markeredgecolor='black', markeredgewidth=2, label='Best performing')

        ax.axhline(y=best_y, color='gold', linestyle='--', alpha=0.7, linewidth=2)

    # Styling
    ax.set_xlabel('Number of agents', fontweight='bold', fontsize=16)
    ax.set_ylabel('Accuracy (%)', fontweight='bold', fontsize=16)
    ax.tick_params(axis='both', labelsize=14)
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.legend(fontsize=12, loc='upper left', frameon=True, fancybox=True, shadow=True,
             framealpha=1.0, facecolor='white', edgecolor='black')
    ax.set_xlim(-0.5, 3.5)
    ax.set_ylim(20, 45)
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels([1, 2, 3, 5])

    # Panel label
    ax.text(-0.02, 1.02, panel_label, transform=ax.transAxes,
            ha='left', va='bottom', fontsize=22, fontweight='bold')

    return ax


def plot_rounds_analysis(ax, df, colors, panel_label='b'):
    """Plot how performance changes with discussion rounds"""

    # Parse experiment names for rounds
    def parse_exp_name_rounds(exp_name):
        parts = exp_name.split('_')
        n_agents = int(parts[0])
        n_rounds = int(parts[2])
        has_search = 'with_search' in exp_name
        return n_agents, n_rounds, has_search

    df[['n_agents', 'n_rounds', 'has_search']] = df['exp_name'].apply(
        lambda x: pd.Series(parse_exp_name_rounds(x))
    )

    # Calculate summary statistics for both search and no-search conditions
    summary_df = df.groupby(['n_agents', 'n_rounds', 'has_search']).agg({
        'accuracy': ['mean', 'std'],
        'avg_time': ['mean', 'std']
    }).round(2)
    summary_df.columns = ['accuracy_mean', 'accuracy_std', 'time_mean', 'time_std']
    summary_df = summary_df.reset_index()

    # Filter for 3 agents with search
    rounds_data_search = summary_df[(summary_df['n_agents'] == 3) & (summary_df['has_search'] == True)]

    if not rounds_data_search.empty:
        # Plot with search line
        ax.plot(rounds_data_search['n_rounds'], rounds_data_search['accuracy_mean'],
               marker='o', linewidth=3, markersize=10,
               color=colors['metrics']['accuracy'], alpha=0.8,
               markeredgecolor='black', markeredgewidth=2, label='With search')

        ax.fill_between(rounds_data_search['n_rounds'],
                       rounds_data_search['accuracy_mean'] - rounds_data_search['accuracy_std'],
                       rounds_data_search['accuracy_mean'] + rounds_data_search['accuracy_std'],
                       color=colors['metrics']['accuracy'], alpha=0.2)

        # Add value annotations
        for i, row in rounds_data_search.iterrows():
            ax.annotate(f'{row["accuracy_mean"]:.1f}%',
                       (row['n_rounds'], row['accuracy_mean']),
                       textcoords="offset points", xytext=(0,15), ha='center',
                       fontsize=12, fontweight='bold', color='black')

    # Filter for 3 agents without search
    rounds_data_no_search = summary_df[(summary_df['n_agents'] == 3) & (summary_df['has_search'] == False)]

    if not rounds_data_no_search.empty:
        # Plot no search line
        ax.plot(rounds_data_no_search['n_rounds'], rounds_data_no_search['accuracy_mean'],
               marker='s', linewidth=3, markersize=10,
               color=colors['metrics']['time'], alpha=0.8,
               markeredgecolor='black', markeredgewidth=2, label='No search')

        ax.fill_between(rounds_data_no_search['n_rounds'],
                       rounds_data_no_search['accuracy_mean'] - rounds_data_no_search['accuracy_std'],
                       rounds_data_no_search['accuracy_mean'] + rounds_data_no_search['accuracy_std'],
                       color=colors['metrics']['time'], alpha=0.2)

        # Add value annotations
        for i, row in rounds_data_no_search.iterrows():
            ax.annotate(f'{row["accuracy_mean"]:.1f}%',
                       (row['n_rounds'], row['accuracy_mean']),
                       textcoords="offset points", xytext=(0,-20), ha='center',
                       fontsize=12, fontweight='bold', color='black')

    # Highlight best performing configuration
    if not summary_df[(summary_df['n_agents'] == 3)].empty:
        best_accuracy_idx = summary_df[(summary_df['n_agents'] == 3)]['accuracy_mean'].idxmax()
        best_point = summary_df.loc[best_accuracy_idx]
        best_x = best_point['n_rounds']
        best_y = best_point['accuracy_mean']
        ax.plot(best_x, best_y, marker='*', markersize=15, color='gold',
               markeredgecolor='black', markeredgewidth=2, label='Best performing')

        ax.axhline(y=best_y, color='gold', linestyle='--', alpha=0.7, linewidth=2)

    # Styling
    ax.set_xlabel('Discussion rounds', fontweight='bold', fontsize=16)
    ax.set_ylabel('Accuracy (%)', fontweight='bold', fontsize=16)
    ax.tick_params(axis='both', labelsize=14)
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.legend(fontsize=12, loc='upper left', frameon=True, fancybox=True, shadow=True,
             framealpha=1.0, facecolor='white', edgecolor='black')
    ax.set_xlim(0.5, 3.5)
    ax.set_ylim(20, 45)
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels([1, 2, 3])

    # Panel label
    ax.text(-0.02, 1.02, panel_label, transform=ax.transAxes,
            ha='left', va='bottom', fontsize=22, fontweight='bold')

    return ax


def plot_enhanced_orchestration_comparison(ax, orchestration_df, colors, panel_label='c'):
    """Plot orchestration styles with role-play vs non-role-play comparison"""

    if orchestration_df.empty:
        orchestration_df = pd.DataFrame({
            'exp_name': ['group_chat_with_orchestrator', 'group_chat_voting_only', 'independent', 'one_on_one_sync',
                        'group_chat_with_orchestrator_disable_role_play', 'group_chat_voting_only_disable_role_play',
                        'independent_disable_role_play', 'one_on_one_sync_disable_role_play'],
            'accuracy': [36.3, 33.0, 29.0, 33.7, 32.7, 29.0, 26.0, 30.0]
        })

    orchestration_df['has_role_play'] = ~orchestration_df['exp_name'].str.contains('disable_role_play')
    orchestration_df['base_name'] = orchestration_df['exp_name'].str.replace('_disable_role_play', '')

    orch_summary = orchestration_df.groupby(['base_name', 'has_role_play']).agg({
        'accuracy': ['mean', 'std']
    }).round(2)

    orch_summary.columns = ['accuracy_mean', 'accuracy_std']
    orch_summary = orch_summary.reset_index()

    order_map = {
        'group_chat_with_orchestrator': 0,
        'group_chat_voting_only': 1,
        'independent': 2,
        'one_on_one_sync': 3
    }
    orch_summary['order'] = orch_summary['base_name'].map(order_map)
    orch_summary = orch_summary.sort_values(['order', 'has_role_play'], ascending=[True, False]).reset_index(drop=True)

    labels = ['Moderated\n Meeting', 'Unmoderated\nChat', 'Independent\nVote', 'Individual\nMeeting']

    role_play_data = orch_summary[orch_summary['has_role_play'] == True]
    no_role_play_data = orch_summary[orch_summary['has_role_play'] == False]

    x_pos = np.arange(len(labels))

    ax.plot(x_pos, role_play_data['accuracy_mean'],
            marker='o', linewidth=3, markersize=10,
            color=colors['metrics']['accuracy'], alpha=0.8,
            markeredgecolor='black', markeredgewidth=2, label='With Role-play')

    ax.fill_between(x_pos,
                    role_play_data['accuracy_mean'] - role_play_data['accuracy_std'],
                    role_play_data['accuracy_mean'] + role_play_data['accuracy_std'],
                    color=colors['metrics']['accuracy'], alpha=0.2)

    ax.plot(x_pos, no_role_play_data['accuracy_mean'],
            marker='s', linewidth=3, markersize=10,
            color=colors['metrics']['time'], alpha=0.8,
            markeredgecolor='black', markeredgewidth=2, label='Without Role-play')

    ax.fill_between(x_pos,
                    no_role_play_data['accuracy_mean'] - no_role_play_data['accuracy_std'],
                    no_role_play_data['accuracy_mean'] + no_role_play_data['accuracy_std'],
                    color=colors['metrics']['time'], alpha=0.2)

    for i, (rp_acc, no_rp_acc) in enumerate(zip(role_play_data['accuracy_mean'], no_role_play_data['accuracy_mean'])):
        ax.annotate(f'{rp_acc:.1f}%',
                   (x_pos[i], rp_acc),
                   textcoords="offset points", xytext=(0,15), ha='center',
                   fontsize=12, fontweight='bold', color='black')
        ax.annotate(f'{no_rp_acc:.1f}%',
                   (x_pos[i], no_rp_acc),
                   textcoords="offset points", xytext=(0,-25), ha='center',
                   fontsize=12, fontweight='bold', color='black')

    best_rp_idx = role_play_data['accuracy_mean'].idxmax()
    best_rp_accuracy = role_play_data['accuracy_mean'].iloc[best_rp_idx]
    ax.plot(x_pos[best_rp_idx], best_rp_accuracy, marker='*', markersize=15, color='gold',
            markeredgecolor='black', markeredgewidth=2,
            zorder=10, label='Best performing')

    ax.axhline(y=best_rp_accuracy, color='gold', linestyle='--', alpha=0.7, linewidth=2)

    ax.set_xlabel('Discussion Strategy', fontweight='bold', fontsize=16)
    ax.set_ylabel('Accuracy (%)', fontweight='bold', fontsize=16)
    ax.tick_params(axis='both', labelsize=14)
    ax.grid(True, alpha=0.2, axis='y', linewidth=0.8, linestyle='-')
    ax.set_ylim(25, 45)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=13)

    ax.legend(fontsize=12, loc='upper left', frameon=True, fancybox=True, shadow=True,
              framealpha=1.0, facecolor='white', edgecolor='black')

    ax.text(-0.02, 1.02, panel_label, transform=ax.transAxes,
            ha='left', va='bottom', fontsize=22, fontweight='bold')

    return ax


def plot_vote_convergence(ax, df, panel_label=None):
    """
    Plot vote entropy convergence across discussion rounds.
    Shows how consensus emerges over multiple rounds of discussion.
    """

    try:
        colors = get_figure_1_colors()

        manchester_colors = get_manchester_colors()
        mode_colors = {
            'group_chat_with_orchestrator': colors['metrics']['accuracy'],
            'group_chat_voting_only': manchester_colors['persimmon'],
            'independent': manchester_colors['black'],
            'one_on_one_sync': manchester_colors['sandy_brown']
        }

        mode_names = {
            'group_chat_with_orchestrator': 'Moderated Team Meeting',
            'group_chat_voting_only': 'Unmoderated Group Chat',
            'independent': 'Independent Vote',
            'one_on_one_sync': 'Individual Meeting'
        }

        for mode in df['discussion_mode'].unique():
            mode_data = df[df['discussion_mode'] == mode].sort_values('round')

            if len(mode_data) > 0:
                color = mode_colors.get(mode, 'gray')
                label = mode_names.get(mode, mode)

                ax.plot(
                    mode_data['round'],
                    mode_data['mean'],
                    label=label,
                    color=color,
                    marker='o',
                    linewidth=3,
                    markersize=10,
                    markeredgecolor='black',
                    markeredgewidth=2,
                    alpha=0.8
                )

                ax.fill_between(
                    mode_data['round'],
                    mode_data['mean'] - mode_data['std'] * 0.05,
                    mode_data['mean'] + mode_data['std'] * 0.05,
                    color=color,
                    alpha=0.2,
                    linewidth=0
                )

        for round_num in df['round'].unique():
            round_data = df[df['round'] == round_num]
            if not round_data.empty:
                best_idx = round_data['mean'].idxmin()
                worst_idx = round_data['mean'].idxmax()

                best_round_data = round_data.loc[best_idx]
                worst_round_data = round_data.loc[worst_idx]

                ax.annotate(f'{best_round_data["mean"]:.2f}',
                           (best_round_data['round'], best_round_data['mean']),
                           textcoords="offset points", xytext=(0,15), ha='center',
                           fontsize=12, fontweight='bold', color='black')

                ax.annotate(f'{worst_round_data["mean"]:.2f}',
                           (worst_round_data['round'], worst_round_data['mean']),
                           textcoords="offset points", xytext=(0,15), ha='center',
                           fontsize=12, fontweight='bold', color='black')

                label = 'Best Consensus' if round_num == sorted(df['round'].unique())[0] else None
                ax.plot(best_round_data['round'], best_round_data['mean'],
                       marker='*', markersize=15, color='gold',
                       markeredgecolor='black', markeredgewidth=2,
                       label=label, zorder=10)

        ax.set_xlabel('Discussion rounds', fontsize=16, fontweight='bold')
        ax.set_ylabel('Vote Entropy (bits)', fontsize=16, fontweight='bold')
        ax.tick_params(axis='both', labelsize=14)

        rounds = sorted(df['round'].unique())
        ax.set_xticks(rounds)
        ax.set_xticklabels([r + 1 for r in rounds])

        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)

        ax.legend(loc='lower left', fontsize=12, frameon=True, fancybox=True, shadow=True,
                 framealpha=1.0, facecolor='white', edgecolor='black')

        ax.set_ylim(0.1, 0.6)

        if panel_label:
            ax.text(-0.02, 1.02, panel_label, transform=ax.transAxes,
                   fontsize=22, fontweight='bold', va='bottom', ha='left')

        return ax

    except FileNotFoundError:
        ax.text(0.5, 0.5, 'Vote Convergence Analysis\n(Data not available)',
               transform=ax.transAxes, ha='center', va='center',
               fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))

        if panel_label:
            ax.text(-0.1, 1.05, panel_label, transform=ax.transAxes,
                   fontsize=22, fontweight='bold', va='bottom', ha='right')

        return ax


def plot_cross_talk_dag(ax, crosstalk_df, colors, panel_label='f'):
    """Plot cross-talk directed acyclic graph showing communication patterns between agents"""

    agents = list(crosstalk_df.columns)
    n_agents = len(agents)

    # Left to right layout with larger horizontal spacing: triage -> 3 experts -> moderator -> 3 experts -> moderator (2nd round) -> 3 experts -> moderator (3rd round) -> vote
    positions = {
        'triage': (0.05, 0.5),
        'expert1': (0.15, 0.7),
        'expert2': (0.15, 0.5),
        'expert3': (0.15, 0.3),
        'moderator': (0.3, 0.5),
        'expert4': (0.45, 0.7),  # Second round experts
        'expert5': (0.45, 0.5),
        'expert6': (0.45, 0.3),
        'moderator2': (0.6, 0.5),  # Second round moderator
        'expert7': (0.75, 0.7),  # Third round experts
        'expert8': (0.75, 0.5),
        'expert9': (0.75, 0.3),
        'moderator3': (0.85, 0.5),  # Third round moderator
        'vote': (0.95, 0.5)  # Final voting stage
    }

    node_colors = {
        'moderator': colors['agents']['moderator'],
        'moderator2': colors['agents']['moderator'],
        'moderator3': colors['agents']['moderator'],
        'triage': colors['agents']['triage'],
        'expert1': colors['agents']['expert1'],
        'expert2': colors['agents']['expert2'],
        'expert3': colors['agents']['expert3'],
        'expert4': colors['agents']['expert1'],  # Reuse colors for second round
        'expert5': colors['agents']['expert2'],
        'expert6': colors['agents']['expert3'],
        'expert7': colors['agents']['expert1'],  # Reuse colors for third round
        'expert8': colors['agents']['expert2'],
        'expert9': colors['agents']['expert3'],
        'vote': colors['component_breakdown']['Final']
    }

    node_radius = 0.03

    # Draw nodes
    for agent in positions.keys():
        x, y = positions[agent]
        circle = plt.Circle((x, y), node_radius, facecolor=node_colors[agent],
                          alpha=0.9, zorder=10, linewidth=2, edgecolor='black')
        ax.add_patch(circle)

        # Add labels for all nodes without values
        if agent == 'vote':
            ax.text(x, y, 'VOTE', ha='center', va='center', fontsize=5, fontweight='bold',
                   color='white', zorder=11)
        elif agent.startswith('expert'):
            expert_num = agent[-1]
            ax.text(x, y, f'E{expert_num}', ha='center', va='center', fontsize=6, fontweight='bold',
                   color='white', zorder=11)
        elif agent == 'moderator3':
            ax.text(x, y, 'M3', ha='center', va='center', fontsize=6, fontweight='bold',
                   color='white', zorder=11)
        elif agent == 'moderator2':
            ax.text(x, y, 'M2', ha='center', va='center', fontsize=6, fontweight='bold',
                   color='white', zorder=11)
        elif agent == 'moderator':
            ax.text(x, y, 'M1', ha='center', va='center', fontsize=6, fontweight='bold',
                   color='white', zorder=11)
        elif agent == 'triage':
            ax.text(x, y, 'T', ha='center', va='center', fontsize=6, fontweight='bold',
                   color='white', zorder=11)

    # Define DAG flow: triage -> experts (round 1) -> moderator -> experts (round 2) -> moderator2 -> experts (round 3) -> moderator3 -> vote
    dag_connections = [
        ('triage', 'expert1'),
        ('triage', 'expert2'),
        ('triage', 'expert3'),
        ('expert1', 'moderator'),
        ('expert2', 'moderator'),
        ('expert3', 'moderator'),
        ('moderator', 'expert4'),
        ('moderator', 'expert5'),
        ('moderator', 'expert6'),
        ('expert4', 'moderator2'),
        ('expert5', 'moderator2'),
        ('expert6', 'moderator2'),
        ('moderator2', 'expert7'),
        ('moderator2', 'expert8'),
        ('moderator2', 'expert9'),
        ('expert7', 'moderator3'),
        ('expert8', 'moderator3'),
        ('expert9', 'moderator3'),
        ('moderator3', 'vote')
    ]

    # Draw DAG arrows (main flow)
    for source, target in dag_connections:
        if source in positions and target in positions:
            x1, y1 = positions[source]
            x2, y2 = positions[target]

            dx = x2 - x1
            dy = y2 - y1
            length = np.sqrt(dx**2 + dy**2)

            start_offset = node_radius + 0.01
            end_offset = node_radius + 0.01

            x1_adj = x1 + (dx / length) * start_offset
            y1_adj = y1 + (dy / length) * start_offset
            x2_adj = x2 - (dx / length) * end_offset
            y2_adj = y2 - (dy / length) * end_offset

            # Main flow arrows (thicker, darker)
            ax.annotate('', xy=(x2_adj, y2_adj), xytext=(x1_adj, y1_adj),
                      arrowprops=dict(arrowstyle='->', lw=2.5,
                                    color=colors['agents']['moderator'], alpha=0.8,
                                    shrinkA=0, shrinkB=0))

    # Draw cross-talk connections (lighter, thinner) - only for original agents in data
    original_agents = [agent for agent in agents if agent in positions]
    for source in original_agents:
        for target in original_agents:
            if source != target and source in positions and target in positions:
                # Skip main DAG connections
                if (source, target) in dag_connections:
                    continue

                weight = crosstalk_df.loc[source, target] if source in crosstalk_df.index and target in crosstalk_df.columns else 0
                if weight > 0:  # Only show non-zero cross-talk
                    x1, y1 = positions[source]
                    x2, y2 = positions[target]

                    dx = x2 - x1
                    dy = y2 - y1
                    length = np.sqrt(dx**2 + dy**2)

                    start_offset = node_radius + 0.01
                    end_offset = node_radius + 0.01

                    x1_adj = x1 + (dx / length) * start_offset
                    y1_adj = y1 + (dy / length) * start_offset
                    x2_adj = x2 - (dx / length) * end_offset
                    y2_adj = y2 - (dy / length) * end_offset

                    # Cross-talk arrows (thinner, curved)
                    connectionstyle = "arc3,rad=0.2"
                    ax.annotate('', xy=(x2_adj, y2_adj), xytext=(x1_adj, y1_adj),
                              arrowprops=dict(arrowstyle='->', lw=1.0,
                                            color=colors['agents']['triage'], alpha=0.6,
                                            shrinkA=0, shrinkB=0,
                                            connectionstyle=connectionstyle,
                                            linestyle='--'))

    # Add stage labels with larger horizontal spacing
    ax.text(0.05, 0.1, 'Triage', ha='center', va='center', fontsize=8, fontweight='bold')
    ax.text(0.15, 0.1, 'Expert\nRound 1', ha='center', va='center', fontsize=8, fontweight='bold')
    ax.text(0.3, 0.1, 'Moderator\nRound 1', ha='center', va='center', fontsize=8, fontweight='bold')
    ax.text(0.45, 0.1, 'Expert\nRound 2', ha='center', va='center', fontsize=8, fontweight='bold')
    ax.text(0.6, 0.1, 'Moderator\nRound 2', ha='center', va='center', fontsize=8, fontweight='bold')
    ax.text(0.75, 0.1, 'Expert\nRound 3', ha='center', va='center', fontsize=8, fontweight='bold')
    ax.text(0.85, 0.1, 'Moderator\nRound 3', ha='center', va='center', fontsize=8, fontweight='bold')
    ax.text(0.95, 0.1, 'Final\nVote', ha='center', va='center', fontsize=8, fontweight='bold')

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    # Legend
    legend_elements = []
    agent_labels = {
        'triage': 'Triage Agent',
        'expert1': 'Expert Agents',
        'moderator': 'Moderator',
        'vote': 'Final Vote'
    }

    # Simplified legend to avoid redundancy
    for agent in ['triage', 'expert1', 'moderator', 'vote']:
        if agent in node_colors:
            legend_elements.append(plt.Line2D([0], [0], marker='o', color='w',
                                            markerfacecolor=node_colors[agent],
                                            markersize=8, label=agent_labels[agent],
                                            markeredgecolor='black', markeredgewidth=1))

    # Add arrow legend
    legend_elements.append(plt.Line2D([0], [0], color=colors['agents']['moderator'], lw=2.5, label='Main Flow'))
    legend_elements.append(plt.Line2D([0], [0], color=colors['agents']['triage'], lw=1.0, linestyle='--', label='Cross-talk'))

    ax.legend(handles=legend_elements, fontsize=9, loc='upper center', ncol=3,
             frameon=True, fancybox=True, shadow=True,
             framealpha=1.0, facecolor='white', edgecolor='black',
             bbox_to_anchor=(0.5, 0.98))

    ax.text(-0.02, 1.02, panel_label, transform=ax.transAxes,
            ha='left', va='bottom', fontsize=22, fontweight='bold')

    return ax


def create_figure_1():
    """Create Figure 1: Agent Configuration Analysis"""

    apply_medagents_style()

    colors = get_figure_1_colors()


    base_dir = os.path.dirname(__file__)
    df = pd.read_csv(os.path.join(base_dir, 'data', 'agent_configuration.csv'))
    print(f"Loaded {len(df)} agent configuration records from real data")

    orchestration_df = pd.read_csv(os.path.join(base_dir, 'data', 'orchestration_style.csv'))
    print(f"Loaded {len(orchestration_df)} orchestration records from real data")

    vote_entropy_df = pd.read_csv(os.path.join(base_dir, 'data', 'vote_entropy_summary.csv'))
    print(f"Loaded {len(vote_entropy_df)} vote entropy records from real data")

    crosstalk_df = pd.read_csv(os.path.join(base_dir, 'data', 'crosstalk.csv'), index_col=0)
    print(f"Loaded {crosstalk_df.shape} crosstalk matrix from real data")

    fig = plt.figure(figsize=(18, 20))
    gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 2], width_ratios=[1, 1, 1],
                          hspace=0.3, wspace=0.2)

    ax1 = fig.add_subplot(gs[0, 0])
    plot_agent_scaling_analysis(ax1, df, colors, panel_label='a')

    ax2 = fig.add_subplot(gs[0, 1])
    plot_rounds_analysis(ax2, df, colors, panel_label='b')

    ax7 = fig.add_subplot(gs[0, 2])
    plot_enhanced_orchestration_comparison(ax7, orchestration_df, colors, panel_label='c')

    ax8 = fig.add_subplot(gs[1, 0])
    plot_vote_convergence(ax8, vote_entropy_df, panel_label='d')

    ax6 = fig.add_subplot(gs[1, 1:])
    plot_cross_talk_dag(ax6, crosstalk_df, colors, panel_label='f')

    ax3 = fig.add_subplot(gs[2, :])
    plot_pdf(ax3, 'discussion_patterns.pdf', panel_label='e')


    plt.tight_layout()
    save_figure(fig, 'fig-3.agent_configuration_analysis.pdf')

    return fig


if __name__ == "__main__":
    create_figure_1()
