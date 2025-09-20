import os
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import ConfusionMatrixDisplay

# Configurazione stile per tesi
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'axes.linewidth': 0.8,
    'grid.linewidth': 0.5,
    'lines.linewidth': 1.5,
})

def plot_confusion_matrix(tp, fp, fn, tn, 
                         title="Matrice di Confusione",
                         show=True, out_dir='figures', filename='confusion_matrix.png',
                         figsize=(6, 5), fontsize=12,
                         cmap='Blues', show_percentages=True):
    
    
    confusion_matrix = np.array([[tn, fp],    # Riga 0: Classe reale Negativa
                                [fn, tp]])    # Riga 1: Classe reale Positiva
    
    # Nomi delle classi di default
    class_names = ['1', '0']
    
    # Crea la figura con le dimensioni specificate
    fig, ax = plt.subplots(figsize=figsize)
    
    # Crea ConfusionMatrixDisplay
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix,
                                  display_labels=class_names)
    
    # Plotta con stile personalizzato
    disp.plot(ax=ax, cmap=cmap, values_format='d',
              colorbar=True, xticks_rotation='horizontal')
    
    # Personalizzazioni per mantenere stile coerente
    ax.set_title(title, fontsize=fontsize + 2, fontweight='bold', pad=20)
    ax.set_xlabel('Predetto', fontsize=fontsize + 1,)
    ax.set_ylabel('Reale', fontsize=fontsize + 1)
    
    # Personalizza font delle etichette
    ax.tick_params(axis='both', labelsize=fontsize)
    
    # Personalizza i valori nella matrice
    for text in disp.text_.ravel():
        if text is not None:
            text.set_fontsize(fontsize)
            text.set_fontweight('bold')
    
    # Personalizza colorbar
    if disp.im_ is not None:
        cbar = disp.im_.colorbar
        if cbar is not None:
            cbar.ax.set_ylabel('Frequenza', rotation=270, labelpad=20,
                              fontsize=fontsize, fontweight='bold')
            cbar.ax.tick_params(labelsize=fontsize-2)
    
    # Rimuovi spines superiori e destri per coerenza
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Calcola e mostra statistiche
    total = tp + fp + fn + tn
    if total > 0:
        accuracy = (tp + tn) / total
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        stats_text = f'Accuratezza: {accuracy:.3f} | Precisione: {precision:.3f} | Richiamo: {recall:.3f} | F1: {f1:.3f}'
        fig.text(0.5, 0.02, stats_text, ha='center', va='bottom', 
                fontsize=fontsize-2, style='italic', color='gray')
    
    plt.tight_layout()
    
    # Aggiusta layout per le statistiche
    if total > 0:
        plt.subplots_adjust(bottom=0.15)
    
    
    os.makedirs(out_dir, exist_ok=True)
    filepath = os.path.join(out_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Matrice di confusione salvata: {filepath}")

    if show:
        plt.show()



    plt.close(fig)
    return filepath
    




def pretty_plot_model_component_bpp(data_dict, model, component, bpp,
                                    save=True, show=True, out_dir='plots',
                                    figsize=(10,5),
                                    fontsize=12,
                                    alpha=0.85,
                                    show_values=True,
                                    value_fmt='{:.3f}',
                                    rotate=30,
                                    value_offset=0.015):
    """
    Plot con valori sopra ogni barra - Stile tesi migliorato.
    data_dict[model][bpp][component][method] = {'y':..., 'y_hat':...}
    """
    mdl = str(model)
    comp = str(component)
    bpp_key = bpp

    if mdl not in data_dict:
        raise ValueError(f"Model {model} non trovato")
    if bpp_key not in data_dict[mdl]:
        raise ValueError(f"bpp {bpp} non trovato per il modello {model}")
    if comp not in data_dict[mdl][bpp_key]:
        raise ValueError(f"Componente {component} non trovata per il modello {model} a {bpp} bpp")

    # estraggo e ordino i metodi
    methods = sorted(list(data_dict[mdl][bpp_key][comp].keys()))
    y_vals = np.array([data_dict[mdl][bpp_key][comp][m].get('y', np.nan) for m in methods], dtype=float)
    yhat_vals = np.array([data_dict[mdl][bpp_key][comp][m].get('y_hat', np.nan) for m in methods], dtype=float)

    # filtro metodi dove entrambi NaN
    mask = ~(np.isnan(y_vals) & np.isnan(yhat_vals))
    methods = [m for m, mk in zip(methods, mask) if mk]
    y_vals = y_vals[mask]
    yhat_vals = yhat_vals[mask]

    x = np.arange(len(methods))
    width = 0.28  # Ridotta da 0.36

    fig, ax = plt.subplots(figsize=figsize)

    # Colori professionali
    color_y = '#2E86AB'      # Blu elegante
    color_yhat = '#A23B72'   # Magenta/viola

    # Barre senza rigature (hatch) e senza bordi neri spessi
    bars_y = ax.bar(x - width/2, y_vals, width,
                    label='y', alpha=alpha,
                    color=color_y, edgecolor='white', linewidth=0.8)
    
    bars_yhat = ax.bar(x + width/2, yhat_vals, width,
                       label='y_hat', alpha=alpha,
                       color=color_yhat, edgecolor='white', linewidth=0.8)

    # regolo ylim per non tagliare le annotazioni
    max_h = 0.0
    if len(y_vals):
        max_h = max(max_h, np.nanmax(y_vals))
    if len(yhat_vals):
        max_h = max(max_h, np.nanmax(yhat_vals))
    top_margin = max(0.06, max_h * 0.12)
    ax.set_ylim(0, min(1.0, max(1.0, max_h + top_margin)))

    # Etichette e titolo
    ax.set_xticks(x)
    ax.set_xticklabels([m.capitalize() for m in methods], 
                       rotation=rotate, ha='right', fontsize=fontsize)
    ax.set_ylabel('Accuratezza', fontsize=fontsize + 1, fontweight='bold')
    
    # Titolo più elegante
    model_names = {'GB': 'Gradient Boosting', 'RF': 'Random Forest'}
    component_names = {'Y': 'Y', 'YUV': 'YUV'}
    model_full = model_names.get(mdl, mdl)
    comp_full = component_names.get(comp, comp)
    
    ax.set_title(f'{model_full} - {comp_full} ({bpp} bpp)', 
                fontsize=fontsize + 2, fontweight='bold', pad=20)

    # Griglia più sottile
    ax.grid(axis='y', linestyle='-', linewidth=0.3, alpha=0.6)
    ax.set_axisbelow(True)
    
    # Rimuovi spines superiori e destri
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)
    
    # Legenda migliorata
    ax.legend(frameon=True, fontsize=fontsize, loc='upper right',
              fancybox=True, shadow=True, framealpha=0.95,
              edgecolor='gray', facecolor='white')

    # annotazioni valori sopra le barre
    if show_values:
        def annotate_bars(bars, fmt):
            for bar in bars:
                h = bar.get_height()
                if np.isnan(h):
                    continue
                x_pos = bar.get_x() + bar.get_width()/2
                y_pos = h + value_offset
                
                # se y_pos supera il limite superiore
                ylim_top = ax.get_ylim()[1]
                if y_pos > ylim_top - 0.02:
                    y_pos = h - value_offset * 1.5
                    va = 'top'
                    color = 'white'
                    weight = 'bold'
                else:
                    va = 'bottom'
                    color = 'black'
                    weight = 'normal'
                    
                ax.text(x_pos, y_pos, fmt.format(h), 
                       ha='center', va=va, 
                       fontsize=fontsize-1, 
                       color=color, fontweight=weight)

        annotate_bars(bars_y, value_fmt)
        annotate_bars(bars_yhat, value_fmt)

    plt.tight_layout()

    path = None
    if save:
        os.makedirs(out_dir, exist_ok=True)
        fname = f"{mdl}_{comp}_{bpp}bpp_improved.png".replace(' ', '_')
        path = os.path.join(out_dir, fname)
        plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Plot salvato: {path}")
    if show:
        plt.show()
    plt.close(fig)
    return path


def compare_models_plot(data_dict, component, bpp,
                        save=True, show=True, out_dir='plots',
                        figsize=(12, 6), fontsize=12,
                        value_fmt='{:.3f}'):
    """
    Confronta GB e RF nello stesso grafico - Stile tesi migliorato.
    X = metodi, per ogni metodo 4 barre: GB-y, GB-y_hat, RF-y, RF-y_hat
    """
    models = ['GB', 'RF']
    methods = sorted(list(
        set(data_dict['GB'][bpp][component].keys())
        | set(data_dict['RF'][bpp][component].keys())
    ))

    x = np.arange(len(methods))
    width = 0.15  # Ridotta da 0.2

    fig, ax = plt.subplots(figsize=figsize)

    # Colori eleganti e coerenti
    colors = {
        ('GB', 'y'): '#2E86AB',      
        ('GB', 'y_hat'): '#7BB3DB',  
        ('RF', 'y'): '#A23B72',      
        ('RF', 'y_hat'): '#D67BA8', 
    }
    
    offsets = {
        ('GB', 'y'): -1.5 * width,
        ('GB', 'y_hat'): -0.5 * width,
        ('RF', 'y'): +0.5 * width,
        ('RF', 'y_hat'): +1.5 * width,
    }

    bars = []
    labels = []

    # Nomi più descrittivi per la legenda
    label_names = {
        ('GB', 'y'): 'GB - y',
        ('GB', 'y_hat'): 'GB - y_hat',
        ('RF', 'y'): 'RF - y',
        ('RF', 'y_hat'): 'RF - y_hat',
    }

    for model in models:
        for target in ['y', 'y_hat']:
            vals = []
            for m in methods:
                if m in data_dict[model][bpp][component]:
                    v = data_dict[model][bpp][component][m].get(target, np.nan)
                else:
                    v = np.nan
                vals.append(v)
            vals = np.array(vals, dtype=float)
            mask = ~np.isnan(vals)
            
            if np.any(mask):  # Solo se ci sono valori validi
                bar = ax.bar(
                    x[mask] + offsets[(model, target)],
                    vals[mask], width,
                    label=label_names[(model, target)],
                    color=colors[(model, target)],
                    edgecolor='white', linewidth=0.8,
                    alpha=0.85
                )
                bars.append(bar)
                labels.append(label_names[(model, target)])

                # annotazioni valori
                for bx, v in zip(x[mask] + offsets[(model, target)], vals[mask]):
                    ax.text(bx, v + 0.015, value_fmt.format(v),
                            ha='center', va='bottom', 
                            fontsize=fontsize-2, fontweight='normal')

    # Configurazione assi
    ax.set_xticks(x)
    ax.set_xticklabels([m.capitalize() for m in methods], 
                       rotation=30, ha='right', fontsize=fontsize)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Accuratezza", fontsize=fontsize + 1, fontweight='bold')
    
    # Titolo più descrittivo
    component_names = {'Y': 'Y', 'YUV': 'YUV'}
    comp_full = component_names.get(component, component)
    ax.set_title(f"Confronto Modelli - {comp_full} ({bpp} bpp)", 
                fontsize=fontsize + 2, fontweight='bold', pad=20)

    # Griglia e stile
    ax.grid(axis='y', linestyle='-', linewidth=0.3, alpha=0.6)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)
    
    # Legenda migliorata con organizzazione a 2 colonne
    ax.legend(frameon=True, fontsize=fontsize-1, 
              loc='upper right', ncol=2,
              fancybox=True, shadow=True, framealpha=0.95,
              edgecolor='gray', facecolor='white',
              columnspacing=1.5)

    plt.tight_layout()

    path = None
    if save:
        os.makedirs(out_dir, exist_ok=True)
        fname = f"Compare_GB_RF_{component}_{bpp}bpp_improved.png"
        path = os.path.join(out_dir, fname)
        plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Plot salvato: {path}")
    if show:
        plt.show()
    plt.close(fig)
    return path



def generate_all_thesis_plots(data_dict, out_dir='figures'):
    """
    Genera tutti i grafici necessari per la tesi con nomenclatura consistente.
    
    Parameters:
    -----------
    data_dict : dict
        Dizionario con struttura [model][bpp][component][method] = {'y':..., 'y_hat':...}
    out_dir : str
        Directory di output per salvare i grafici
        
    Returns:
    --------
    list : Lista dei file generati
    """
    
    print(" Generazione grafici per tesi triennale...")
    print("=" * 60)
    
    generated_files = []
    
    # Grafici individuali per ogni modello
    print("\n Grafici individuali per modello:")
    for model in ['GB', 'RF']:
        model_name = {'GB': 'Gradient Boosting', 'RF': 'Random Forest'}[model]
        
        for bpp in [6, 12]:
            for component in ['Y', 'YUV']:
                comp_name = {'Y': 'Luminanza', 'YUV': 'Crominanza'}[component]
                
                try:
                    print(f"  • {model_name} - {comp_name} - {bpp} bpp", end="")
                    
                    filepath = pretty_plot_model_component_bpp(
                        data_dict, model, component, bpp,
                        save=True, out_dir=out_dir
                    )
                    
                        
                except Exception as e:
                    print(f"  (errore: {e})")
    
    # Grafici di confronto
    print("\ Grafici di confronto modelli:")
    for bpp in [6, 12]:
        for component in ['Y', 'YUV']:
            comp_name = {'Y': 'Luminanza', 'YUV': 'Crominanza'}[component]
            
            try:
                print(f"  • Confronto GB vs RF - {comp_name} - {bpp} bpp", end="")
                
                filepath = compare_models_plot(
                    data_dict, component, bpp,
                    save=True, out_dir=out_dir
                )
                
                    
            except Exception as e:
                print(f" errore: {e})")
    
    return generated_files
# Esempio di utilizzo
if __name__ == "__main__":
    pass