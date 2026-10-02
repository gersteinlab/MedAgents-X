"""Unified plotting utilities and color schemes for all figures."""
from pathlib import Path
import matplotlib.pyplot as plt


def get_manchester_colors():
    return {
        "black": "#000000",
        "jet": "#444444",
        "gray": "#888888",
        "french_gray": "#BCB9BE",
        "snow": "#FFF0F7",
        "penn_red": "#BB3234",
        "barn_red": "#A40000",
        "engineering_orange": "#D26367",
        "pink": "#FFC5CE",
        "jasmine": "#FFE697",
        "sunset": "#F9C578",
        "sandy_brown": "#F2A358",
        "sandy_brown_light": "#F8D1AB",
        "persimmon": "#F47B3C",
    }


def get_nature_biotechnology_colors():
    return {
        "nature_red": "#E31A1C",
        "nature_blue": "#1F78B4",
        "nature_green": "#33A02C",
        "nature_orange": "#FF7F00",
        "nature_purple": "#6A3D9A",
        "nature_yellow": "#FFFF99",
        "nature_brown": "#B15928",
        "nature_pink": "#FB9A99",
        "nature_light_blue": "#A6CEE3",
        "nature_light_green": "#B2DF8A",
        "nature_light_orange": "#FDBF6F",
        "nature_light_purple": "#CAB2D6",
        "nature_grey": "#999999",
        "nature_dark_red": "#B2182B",
        "nature_dark_blue": "#2166AC",
        "nature_teal": "#35978F",
    }


def get_figure_0_colors():
    mc = get_manchester_colors()
    return {
        'methods': {
            'MedAgents-X': mc['jasmine'],
            'AFlow': mc['sandy_brown_light'],
            'SPO': mc['sunset'],
            'MultiPersona': mc['sandy_brown'],
            'Self-refine': mc['persimmon'],
            'CoT': mc['barn_red'],
            'CoT-SC': mc['penn_red'],
            'MedAgents': mc['pink'],
            'MDAgents': mc['snow'],
            'MedPrompt': mc['french_gray'],
            'MedRAG': mc['gray'],
            'Few-shot': mc['jet'],
            'Zero-shot': mc['black'],
        },
        'metrics': {
            'accuracy': mc['penn_red'],
            'time': mc['jasmine'],
            'cost': mc['black'],
        },
        'annotations': {
            'text_box_face': mc['snow'],
            'text_box_edge': mc['gray'],
            'text': mc['black'],
        },
        'component_breakdown': {
            'Baseline': mc['gray'],
            'Multi-Agent': mc['jet'],
            'Evidence_Retrieval': mc['sandy_brown'],
            'Iterative_Reasoning': mc['sunset'],
            'Role_Play': mc['penn_red'],
            'Discussion_Orchestration': mc['jasmine'],
            'Final': mc['barn_red'],
        },
    }


def get_figure_1_colors():
    mc = get_manchester_colors()
    nb = get_nature_biotechnology_colors()
    return {
        'no_search': mc['jet'],
        'with_search': mc['penn_red'],
        'rounds': {1: mc['penn_red'], 2: mc['sandy_brown'], 3: mc['jasmine']},
        'agents': {
            'moderator': mc['penn_red'],
            'triage': mc['gray'],
            'expert1': mc['sunset'],
            'expert2': mc['jasmine'],
            'expert3': mc['sandy_brown'],
        },
        'architecture': {
            'agent': mc['black'],
            'search': mc['sandy_brown'],
            'reasoning': mc['sunset'],
            'knowledge': mc['penn_red'],
        },
        'metrics': {
            'accuracy': mc['penn_red'],
            'time': mc['sandy_brown'],
            'cost': mc['black'],
        },
        'role_play': {
            'enable_role_play': mc['penn_red'],
            'disable_role_play': mc['gray'],
        },
        'orchestration': {
            'group_chat_with_orchestrator': nb['nature_red'],
            'group_chat_voting_only': nb['nature_blue'],
            'independent': nb['nature_green'],
            'one_on_one_sync': nb['nature_purple'],
        },
        'component_breakdown': {
            'Baseline': mc['gray'],
            'Multi-Agent': mc['jet'],
            'Evidence_Retrieval': mc['sandy_brown'],
            'Iterative_Reasoning': mc['sunset'],
            'Role_Play': mc['penn_red'],
            'Discussion_Orchestration': mc['jasmine'],
            'Final': mc['barn_red'],
        },
    }


def get_figure_3_colors():
    mc = get_manchester_colors()
    return {
        'modality': {
            'both': mc['penn_red'],
            'vector_only': mc['jasmine'],
            'web_only': mc['pink'],
            'none': mc['black'],
            'random': mc['gray'],
        },
        'features': {
            'baseline': mc['penn_red'],
            'no_document_review': mc['sunset'],
            'no_query_rewrite': mc['jet'],
            'no_rewrite_no_review': mc['gray'],
        },
        'history': {
            'individual': mc['penn_red'],
            'shared': mc['sandy_brown'],
        },
        'depth': {
            'more_docs': mc['penn_red'],
        },
        'metrics': {
            'accuracy': mc['penn_red'],
            'time': mc['sandy_brown'],
            'cost': mc['black'],
        },
        'annotations': {
            'text_box_face': mc['snow'],
            'text_box_edge': mc['gray'],
            'text': mc['black'],
        },
    }


def get_background_colors():
    """Get background color options."""
    return {
        'white': '#ffffff',
        'light_gray': '#f8f9fa',
        'cream': '#fafafa',
    }


def apply_standard_plot_formatting(ax, panel_label=None, grid_alpha=0.15, grid_linewidth=0.5, background_color='white', fontsize=22, pad=6):
    """Apply standard formatting to a plot axis with panel labeling."""
    if panel_label is not None:
        label_text = str(panel_label).strip()
        ax.text(-0.02, 1.02, label_text, transform=ax.transAxes,
                ha='left', va='bottom', fontsize=fontsize, fontweight='bold')

    ax.set_facecolor(background_color)
    if grid_alpha and grid_alpha > 0:
        ax.grid(True, alpha=grid_alpha, linewidth=grid_linewidth, axis='y')

    if ax.get_xaxis() is not None and ax.xaxis.label is not None:
        ax.xaxis.label.set_weight('bold')
    if ax.get_yaxis() is not None and ax.yaxis.label is not None:
        ax.yaxis.label.set_weight('bold')


def apply_medagents_style():
    plt.style.use(Path(__file__).with_name('medagents.mplstyle'))


def save_figure(fig, filename):
    output = Path(__file__).resolve().parents[1] / 'output/figures' / filename
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(output)


def plot_pdf(ax, filename, panel_label):
    from pdf2image import convert_from_path

    path = Path(__file__).parent / 'assets' / filename
    image = convert_from_path(path, dpi=300, first_page=1, last_page=1)[0]
    ax.imshow(image, aspect='equal')
    ax.axis('off')
    ax.text(-0.02, 1.02, panel_label, transform=ax.transAxes,
            ha='left', va='bottom', fontsize=22, fontweight='bold')
    return ax
