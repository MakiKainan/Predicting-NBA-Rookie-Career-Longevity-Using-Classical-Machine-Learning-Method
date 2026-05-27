import { useState } from 'react';
import { useIntersectionObserver } from '../hooks/useIntersectionObserver';
import { Maximize2, X, Cpu, TrendingUp, ShieldAlert, Award, FileText } from 'lucide-react';

interface MetricRow {
  name: string;
  baseline: string;
  tuned: string;
  improvement: string;
  isPositive: boolean;
  description: string;
}

export default function ModelSection() {
  const [ref, isVisible] = useIntersectionObserver<HTMLElement>();
  const [activeTab, setActiveTab] = useState<'logistic' | 'tree'>('logistic');
  const [lightboxImage, setLightboxImage] = useState<{ src: string; title: string } | null>(null);

  // Performance metrics for Logistic Regression
  const logisticMetrics: MetricRow[] = [
    { name: 'Accuracy', baseline: '67.18%', tuned: '69.47%', improvement: '+2.29%', isPositive: true, description: 'Overall rate of correct predictions.' },
    { name: 'Precision', baseline: '71.51%', tuned: '73.45%', improvement: '+1.94%', isPositive: true, description: 'Accuracy of positive career longevity predictions.' },
    { name: 'Recall', baseline: '78.53%', tuned: '79.75%', improvement: '+1.23%', isPositive: true, description: 'Sensitivity (capturing rookies who actually last 5+ years).' },
    { name: 'F1-Score', baseline: '74.85%', tuned: '76.47%', improvement: '+1.62%', isPositive: true, description: 'Harmonic mean balancing precision and recall.' },
    { name: 'ROC-AUC', baseline: '74.36%', tuned: '74.74%', improvement: '+0.38%', isPositive: true, description: 'Discriminative power across probability thresholds.' }
  ];

  // Performance metrics for Decision Tree
  const treeMetrics: MetricRow[] = [
    { name: 'Accuracy', baseline: '64.12%', tuned: '67.18%', improvement: '+3.05%', isPositive: true, description: 'Overall rate of correct predictions.' },
    { name: 'Precision', baseline: '71.43%', tuned: '78.52%', improvement: '+7.09%', isPositive: true, description: 'Accuracy of positive career longevity predictions.' },
    { name: 'Recall', baseline: '70.55%', tuned: '65.03%', improvement: '-5.52%', isPositive: false, description: 'Sensitivity (capturing rookies who actually last 5+ years).' },
    { name: 'F1-Score', baseline: '70.99%', tuned: '71.14%', improvement: '+0.15%', isPositive: true, description: 'Harmonic mean balancing precision and recall.' },
    { name: 'ROC-AUC', baseline: '62.04%', tuned: '71.11%', improvement: '+9.07%', isPositive: true, description: 'Discriminative power across probability thresholds.' }
  ];

  const activeMetrics = activeTab === 'logistic' ? logisticMetrics : treeMetrics;

  return (
    <section id="model" ref={ref} className="py-24 md:py-32 px-6 max-w-7xl mx-auto w-full">
      {/* Title block */}
      <div className={`flex flex-col items-center text-center mb-12 ${isVisible ? 'animate-fade-rise' : 'opacity-0'}`}>
        <h2 className="text-5xl md:text-6xl font-display text-secondary uppercase leading-none mb-6">Model Architecture & Results</h2>
        <div className="w-24 h-2 bg-primary mb-6"></div>
        <p className="text-muted-foreground text-lg max-w-3xl font-body">
          We leverage two distinct machine learning paradigms to evaluate rookie seasons. Toggle below to review the mathematical architectures, hyperparameter configurations, and real training results.
        </p>
      </div>

      {/* Model Selection Tabs */}
      <div className={`flex justify-center gap-4 mb-12 ${isVisible ? 'animate-fade-rise-delay' : 'opacity-0'}`}>
        <button
          onClick={() => setActiveTab('logistic')}
          className={`px-6 py-4 font-display text-xl uppercase tracking-wider border-3 transition-all duration-300 shadow-md ${
            activeTab === 'logistic'
              ? 'bg-secondary text-white border-primary translate-y-0.5 shadow-sm'
              : 'bg-white text-secondary border-muted hover:bg-muted hover:border-secondary'
          }`}
        >
          Logistic Regression
        </button>
        <button
          onClick={() => setActiveTab('tree')}
          className={`px-6 py-4 font-display text-xl uppercase tracking-wider border-3 transition-all duration-300 shadow-md ${
            activeTab === 'tree'
              ? 'bg-secondary text-white border-primary translate-y-0.5 shadow-sm'
              : 'bg-white text-secondary border-muted hover:bg-muted hover:border-secondary'
          }`}
        >
          Decision Tree
        </button>
      </div>

      <div className={`grid grid-cols-1 lg:grid-cols-12 gap-8 ${isVisible ? 'animate-fade-rise-delay-2' : 'opacity-0'}`}>
        
        {/* LEFT COLUMN: Architecture & Parameters (5 cols) */}
        <div className="lg:col-span-5 space-y-6 flex flex-col justify-between">
          {/* Architecture Details Card */}
          <div className="bg-white p-8 shadow-xl border-t-8 border-secondary flex-1">
            <div className="flex items-center gap-3 mb-6">
              <Cpu className="w-8 h-8 text-primary" />
              <h3 className="text-3xl font-display text-secondary uppercase">
                {activeTab === 'logistic' ? 'Logistic Regression' : 'Decision Tree'}
              </h3>
            </div>
            
            <div className="space-y-6 text-muted-foreground font-body text-sm leading-relaxed">
              {activeTab === 'logistic' ? (
                <>
                  <p>
                    <strong>Linear Probabilistic Architecture:</strong> The Logistic Regression model fits a sigmoid function over a linear combination of standardized rookie statistics. It directly computes the probability of a player exceeding a 5-year career benchmark.
                  </p>
                  <p>
                    <strong>High Interpretability:</strong> By isolating coefficients ($w_i$), scouts can measure the log-odds impact of every single rookie performance metric, holding other parameters constant.
                  </p>
                </>
              ) : (
                <>
                  <p>
                    <strong>Non-Linear Partitioning Architecture:</strong> The Decision Tree model uses recursive binary splitting to segment rookie profiles. It excels at capturing complex non-linear combinations and threshold interactions (e.g., combinations of low minutes but high productivity).
                  </p>
                  <p>
                    <strong>Hierarchical Rules:</strong> Instead of simple weights, the tree maps out decision rules based on information gain, sorting rookies into terminal nodes indicating career longevity probabilities.
                  </p>
                </>
              )}
            </div>
          </div>

          {/* Hyperparameters Card */}
          <div className="bg-slate-900 text-white p-8 shadow-xl border-l-8 border-primary">
            <h4 className="font-display text-xl uppercase tracking-wider text-primary mb-4">Tuned Hyperparameters</h4>
            {activeTab === 'logistic' ? (
              <ul className="space-y-2 font-mono text-sm text-slate-300">
                <li className="flex justify-between border-b border-slate-800 pb-1">
                  <span>Regularization Penalty:</span>
                  <span className="text-white font-bold">L2 (Ridge)</span>
                </li>
                <li className="flex justify-between border-b border-slate-800 pb-1">
                  <span>Regularization Strength (C):</span>
                  <span className="text-white font-bold">0.1624</span>
                </li>
                <li className="flex justify-between border-b border-slate-800 pb-1">
                  <span>Inverse Strength (&alpha;):</span>
                  <span className="text-white font-bold">&asymp; 6.16</span>
                </li>
                <li className="flex justify-between">
                  <span>Solver algorithm:</span>
                  <span className="text-white font-bold">liblinear</span>
                </li>
              </ul>
            ) : (
              <ul className="space-y-2 font-mono text-sm text-slate-300">
                <li className="flex justify-between border-b border-slate-800 pb-1">
                  <span>Splitting Criterion:</span>
                  <span className="text-white font-bold">Entropy</span>
                </li>
                <li className="flex justify-between border-b border-slate-800 pb-1">
                  <span>Max Depth Limit:</span>
                  <span className="text-white font-bold">5</span>
                </li>
                <li className="flex justify-between border-b border-slate-800 pb-1">
                  <span>Min Samples Split:</span>
                  <span className="text-white font-bold">5</span>
                </li>
                <li className="flex justify-between border-b border-slate-800 pb-1">
                  <span>Min Samples Leaf:</span>
                  <span className="text-white font-bold">2</span>
                </li>
                <li className="flex justify-between">
                  <span>Max Features per Split:</span>
                  <span className="text-white font-bold">sqrt</span>
                </li>
              </ul>
            )}
            <p className="text-[11px] text-slate-400 mt-4 leading-normal font-body">
              Optimized via Stratified 5-Fold Grid Search to maximize testing F1-Score and prevent training-set overfitting.
            </p>
          </div>
        </div>

        {/* RIGHT COLUMN: Real Performance Metrics Table (7 cols) */}
        <div className="lg:col-span-7 bg-muted p-8 md:p-10 shadow-xl border-t-8 border-primary flex flex-col justify-between">
          <div>
            <div className="flex items-center gap-3 mb-6">
              <TrendingUp className="w-8 h-8 text-secondary" />
              <h3 className="text-3xl font-display text-secondary uppercase">Performance Comparison</h3>
            </div>
            
            <div className="overflow-x-auto w-full custom-scrollbar">
              <table className="w-full text-left font-body border-collapse">
                <thead>
                  <tr className="border-b border-slate-300 text-xs font-bold uppercase tracking-widest text-muted-foreground">
                    <th className="py-3 px-2">Metric</th>
                    <th className="py-3 px-2 text-center">Baseline</th>
                    <th className="py-3 px-2 text-center">Tuned</th>
                    <th className="py-3 px-2 text-center">Delta</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-200 text-sm">
                  {activeMetrics.map((row) => (
                    <tr key={row.name} className="hover:bg-slate-100/50 transition-colors duration-150">
                      <td className="py-4 px-2">
                        <div className="font-semibold text-secondary">{row.name}</div>
                        <div className="text-[11px] text-muted-foreground mt-0.5 max-w-xs">{row.description}</div>
                      </td>
                      <td className="py-4 px-2 text-center font-mono font-medium text-muted-foreground">{row.baseline}</td>
                      <td className="py-4 px-2 text-center font-mono font-bold text-slate-900">{row.tuned}</td>
                      <td className="py-4 px-2 text-center">
                        <span className={`inline-block text-xs font-bold font-mono px-2 py-0.5 rounded ${
                          row.isPositive 
                            ? 'bg-emerald-100 text-emerald-800' 
                            : 'bg-rose-100 text-rose-800'
                        }`}>
                          {row.improvement}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          <div className="mt-6 p-4 bg-white/70 border border-slate-200 rounded flex gap-3 text-xs text-muted-foreground leading-normal">
            <Award className="w-5 h-5 text-primary flex-shrink-0 mt-0.5" />
            <p>
              {activeTab === 'logistic' 
                ? 'Applying L2 regularization reduces testing overfitting, raising general accuracy by 2.29% while maintaining a balanced 79.75% recall rate for talent scouting.' 
                : 'Tuning tree depth limits complexity from 18 to 5 levels, pruning 209 leaves down to 27. This drastically reduces the generalization gap while boosting ROC-AUC by 9.07%.'}
            </p>
          </div>
        </div>
      </div>

      {/* Visual Reports Section (Interactive Grid of Charts) */}
      <div className={`mt-16 ${isVisible ? 'animate-fade-rise-delay-2' : 'opacity-0'}`}>
        <div className="flex items-center gap-3 mb-8 border-b-2 border-muted pb-4">
          <FileText className="w-8 h-8 text-primary" />
          <h3 className="text-3xl font-display text-secondary uppercase">Visual Performance Reports</h3>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
          
          {/* Card 1: Confusion Matrix */}
          <div className="bg-white border border-border shadow-md group overflow-hidden flex flex-col justify-between hover:shadow-xl transition-all duration-300">
            <div className="p-6">
              <h4 className="font-display text-lg uppercase tracking-wider text-secondary mb-2">Confusion Matrix</h4>
              <p className="text-xs text-muted-foreground leading-relaxed mb-4">
                Benchmarks the counts of True Positives, True Negatives, False Positives, and False Negatives.
              </p>
            </div>
            <div 
              onClick={() => setLightboxImage({
                src: activeTab === 'logistic' ? '/images/lr_confusion_matrix.png' : '/images/dt_confusion_matrix.png',
                title: `${activeTab === 'logistic' ? 'Logistic Regression' : 'Decision Tree'} - Confusion Matrix`
              })}
              className="relative aspect-video bg-muted border-t border-border overflow-hidden cursor-zoom-in group"
            >
              <img 
                src={activeTab === 'logistic' ? '/images/lr_confusion_matrix.png' : '/images/dt_confusion_matrix.png'} 
                alt="Confusion Matrix" 
                className="w-full h-full object-cover transform group-hover:scale-105 transition-transform duration-500"
              />
              <div className="absolute inset-0 bg-slate-900/0 group-hover:bg-slate-900/40 flex items-center justify-center transition-all duration-300 opacity-0 group-hover:opacity-100">
                <div className="bg-white/95 text-secondary p-3 rounded-full shadow-lg transform translate-y-4 group-hover:translate-y-0 transition-transform duration-300">
                  <Maximize2 className="w-5 h-5" />
                </div>
              </div>
            </div>
          </div>

          {/* Card 2: ROC Curve */}
          <div className="bg-white border border-border shadow-md group overflow-hidden flex flex-col justify-between hover:shadow-xl transition-all duration-300">
            <div className="p-6">
              <h4 className="font-display text-lg uppercase tracking-wider text-secondary mb-2">ROC Curve Comparison</h4>
              <p className="text-xs text-muted-foreground leading-relaxed mb-4">
                Displays the trade-off between Sensitivity (TPR) and Specificity (FPR) across all thresholds.
              </p>
            </div>
            <div 
              onClick={() => setLightboxImage({
                src: activeTab === 'logistic' ? '/images/lr_roc_curve.png' : '/images/dt_roc_curve.png',
                title: `${activeTab === 'logistic' ? 'Logistic Regression' : 'Decision Tree'} - ROC Curve`
              })}
              className="relative aspect-video bg-muted border-t border-border overflow-hidden cursor-zoom-in group"
            >
              <img 
                src={activeTab === 'logistic' ? '/images/lr_roc_curve.png' : '/images/dt_roc_curve.png'} 
                alt="ROC Curve" 
                className="w-full h-full object-cover transform group-hover:scale-105 transition-transform duration-500"
              />
              <div className="absolute inset-0 bg-slate-900/0 group-hover:bg-slate-900/40 flex items-center justify-center transition-all duration-300 opacity-0 group-hover:opacity-100">
                <div className="bg-white/95 text-secondary p-3 rounded-full shadow-lg transform translate-y-4 group-hover:translate-y-0 transition-transform duration-300">
                  <Maximize2 className="w-5 h-5" />
                </div>
              </div>
            </div>
          </div>

          {/* Card 3: Feature Importance / Coefficients */}
          <div className="bg-white border border-border shadow-md group overflow-hidden flex flex-col justify-between hover:shadow-xl transition-all duration-300">
            <div className="p-6">
              <h4 className="font-display text-lg uppercase tracking-wider text-secondary mb-2">
                {activeTab === 'logistic' ? 'Model Coefficients' : 'Feature Importances'}
              </h4>
              <p className="text-xs text-muted-foreground leading-relaxed mb-4">
                {activeTab === 'logistic' 
                  ? 'Standardized log-odds beta values indicating positive or negative impact direction.' 
                  : 'Gini impurity reduction scores indicating feature split frequency and strength.'}
              </p>
            </div>
            <div 
              onClick={() => setLightboxImage({
                src: activeTab === 'logistic' ? '/images/lr_feature_importance.png' : '/images/dt_feature_importance.png',
                title: activeTab === 'logistic' ? 'Logistic Regression - Coefficients' : 'Decision Tree - Feature Importances'
              })}
              className="relative aspect-video bg-muted border-t border-border overflow-hidden cursor-zoom-in group"
            >
              <img 
                src={activeTab === 'logistic' ? '/images/lr_feature_importance.png' : '/images/dt_feature_importance.png'} 
                alt="Feature Importance" 
                className="w-full h-full object-cover transform group-hover:scale-105 transition-transform duration-500"
              />
              <div className="absolute inset-0 bg-slate-900/0 group-hover:bg-slate-900/40 flex items-center justify-center transition-all duration-300 opacity-0 group-hover:opacity-100">
                <div className="bg-white/95 text-secondary p-3 rounded-full shadow-lg transform translate-y-4 group-hover:translate-y-0 transition-transform duration-300">
                  <Maximize2 className="w-5 h-5" />
                </div>
              </div>
            </div>
          </div>

        </div>
      </div>

      {/* Scouting Insights section */}
      <div className={`mt-16 bg-white p-8 md:p-12 shadow-xl border-t-8 border-secondary ${isVisible ? 'animate-fade-rise-delay-2' : 'opacity-0'}`}>
        <div className="flex items-center gap-3 mb-8">
          <ShieldAlert className="w-8 h-8 text-primary" />
          <h3 className="text-3xl font-display text-secondary uppercase font-bold">Sports Analytics Scouting Insights</h3>
        </div>

        {activeTab === 'logistic' ? (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8 text-sm font-body text-muted-foreground leading-relaxed">
            <div className="space-y-4">
              <div>
                <h5 className="font-bold text-secondary text-base uppercase mb-1">1. Three-Point Efficiency Dilemma (3PM vs 3PA)</h5>
                <p>
                  <strong>3PM (Made)</strong> has a massive positive coefficient of <strong>+1.168</strong> (Odds Ratio = <strong>3.21</strong>), multiplying a rookie's 5+ year career odds by <strong>3.21x</strong> per standard deviation. However, <strong>3PA (Attempts)</strong> holds a heavy negative coefficient of <strong>-1.188</strong> (Odds Ratio = <strong>0.30</strong>). High-volume shooting without conversion is highly detrimental to career longevity.
                </p>
              </div>
              <div>
                <h5 className="font-bold text-secondary text-base uppercase mb-1">2. Durability is Built on Games Played (GP)</h5>
                <p>
                  <strong>GP</strong> has a strong positive coefficient of <strong>+0.623</strong> (Odds Ratio = <strong>1.86</strong>). Earning rotation minutes and staying healthy in the rookie year makes a player <strong>86% more likely</strong> to reach the 5-year career threshold.
                </p>
              </div>
            </div>
            <div className="space-y-4">
              <div>
                <h5 className="font-bold text-secondary text-base uppercase mb-1">3. Rebounding & Post Presence (OREB vs DREB)</h5>
                <p>
                  <strong>OREB (Offensive Rebounds)</strong> is a strong positive predictor (<strong>+0.506</strong>, Odds Ratio = <strong>1.66</strong>), representing physical size, hustle, and active second-chance efforts. Defensive rebounds are slightly negative (<strong>-0.154</strong>), as they tend to correlate with baseline minutes rather than independent hustle.
                </p>
              </div>
              <div>
                <h5 className="font-bold text-secondary text-base uppercase mb-1">4. Free Throw Conversion (FTM vs FTA)</h5>
                <p>
                  Similar to the 3-point dilemma, <strong>FTM (Made)</strong> is positive (<strong>+0.480</strong>, Odds Ratio = <strong>1.62</strong>) while <strong>FTA (Attempts)</strong> is negative (<strong>-0.401</strong>, Odds Ratio = <strong>0.67</strong>). Conversions are highly valued, while low-percentage foul line liabilities are penalized.
                </p>
              </div>
            </div>
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8 text-sm font-body text-muted-foreground leading-relaxed">
            <div className="space-y-4">
              <div>
                <h5 className="font-bold text-secondary text-base uppercase mb-1">1. Games Played (GP) is the Primary Split</h5>
                <p>
                  The Decision Tree maps <strong>GP (Games Played)</strong> as the root decision node (importance = <strong>21.78%</strong>). A rookie failing to exceed the threshold is immediately funneled down to low-longevity branches, indicating durability is the fundamental filter.
                </p>
              </div>
              <div>
                <h5 className="font-bold text-secondary text-base uppercase mb-1">2. Non-Linear Efficiency Thresholds</h5>
                <p>
                  The tree finds threshold values where moderate minutes with high offensive/defensive rebound ratios secure roster spots, highlighting that specialized role players have paths to longevity.
                </p>
              </div>
            </div>
            <div className="space-y-4">
              <div>
                <h5 className="font-bold text-secondary text-base uppercase mb-1">3. Top Gini Feature Importances</h5>
                <p>
                  The top 5 predictive variables used in tree construction are <strong>GP (21.78%)</strong>, <strong>OREB (8.08%)</strong>, <strong>DREB (7.42%)</strong>, <strong>PTS (7.16%)</strong>, and <strong>FGA (6.90%)</strong>.
                </p>
              </div>
              <div>
                <h5 className="font-bold text-secondary text-base uppercase mb-1">4. Robust Generalization</h5>
                <p>
                  By limiting tree depth, we successfully balanced the split tree (pruned to 27 leaves, accuracy 67.18% on test). This guarantees the tree evaluates general traits rather than memorizing individual player names.
                </p>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Lightbox Modal overlay */}
      {lightboxImage && (
        <div 
          className="fixed inset-0 bg-slate-950/80 backdrop-blur-md flex flex-col justify-center items-center z-[999] p-4 cursor-zoom-out animate-fade-in"
          onClick={() => setLightboxImage(null)}
        >
          <button 
            className="absolute top-6 right-6 text-white/70 hover:text-white bg-slate-800/80 hover:bg-slate-800 p-3 rounded-full transition-colors duration-200"
            onClick={() => setLightboxImage(null)}
          >
            <X className="w-6 h-6" />
          </button>
          
          <div 
            className="max-w-5xl max-h-[80vh] w-full flex items-center justify-center"
            onClick={(e) => e.stopPropagation()}
          >
            <img 
              src={lightboxImage.src} 
              alt={lightboxImage.title} 
              className="max-w-full max-h-[75vh] object-contain border border-slate-700 shadow-2xl bg-white"
            />
          </div>
          
          <h4 className="text-white text-xl font-display uppercase tracking-wider mt-6 bg-slate-900/85 px-4 py-2 border border-slate-700">
            {lightboxImage.title}
          </h4>
        </div>
      )}
    </section>
  );
}
