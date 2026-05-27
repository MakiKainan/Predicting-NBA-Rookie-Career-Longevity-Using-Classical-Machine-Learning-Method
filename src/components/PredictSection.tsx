import { useState, useEffect, useMemo, useRef } from 'react';
import { Search } from 'lucide-react';
import { useIntersectionObserver } from '../hooks/useIntersectionObserver';

// TypeScript Interfaces for static database and model configurations
interface PlayerStats {
  [key: string]: number | null;
}

interface Player {
  name: string;
  id: number | null;
  stats: PlayerStats;
  actual_5yrs: number;
}

interface ScalerConfig {
  mean: number[];
  scale: number[];
}

interface ImputerConfig {
  medians: number[];
}

interface LogisticRegressionConfig {
  coefficients: number[];
  intercept: number;
}

interface DecisionTreeNode {
  leaf: boolean;
  probabilities?: number[];
  prediction?: number;
  feature_idx?: number;
  threshold?: number;
  left?: DecisionTreeNode;
  right?: DecisionTreeNode;
}

interface ModelConfig {
  features: string[];
  scaler: ScalerConfig;
  imputer: ImputerConfig;
  logistic_regression: LogisticRegressionConfig;
  decision_tree: DecisionTreeNode;
}

// 1. FEATURE METADATA (for labels, ranges, and steps)
const FEATURE_META: Record<string, { label: string; min: number; max: number; step: number }> = {
  "GP": { label: "Games Played (GP)", min: 1, max: 82, step: 1 },
  "MIN": { label: "Minutes per Game (MIN)", min: 1, max: 48, step: 0.1 },
  "PTS": { label: "Points per Game (PTS)", min: 0, max: 35, step: 0.1 },
  "FGS": { label: "Field Goals Made (FGS)", min: 0, max: 15, step: 0.1 },
  "FGA": { label: "Field Goals Attempted (FGA)", min: 0, max: 30, step: 0.1 },
  "3PM": { label: "3-Points Made (3PM)", min: 0, max: 6, step: 0.1 },
  "3PA": { label: "3-Points Attempted (3PA)", min: 0, max: 15, step: 0.1 },
  "FTM": { label: "Free Throws Made (FTM)", min: 0, max: 12, step: 0.1 },
  "FTA": { label: "Free Throws Attempted (FTA)", min: 0, max: 15, step: 0.1 },
  "OREB": { label: "Offensive Rebounds (OREB)", min: 0, max: 8, step: 0.1 },
  "DREB": { label: "Defensive Rebounds (DREB)", min: 0, max: 12, step: 0.1 },
  "AST": { label: "Assists (AST)", min: 0, max: 15, step: 0.1 },
  "STL": { label: "Steals (STL)", min: 0, max: 5, step: 0.1 },
  "BLK": { label: "Blocks (BLK)", min: 0, max: 5, step: 0.1 },
  "TOV": { label: "Turnovers (TOV)", min: 0, max: 6, step: 0.1 }
};


// Helper to normalize player names
const cleanPlayerName = (name: string): string => {
  if (!name) return "";
  return name.endsWith('*') ? name.slice(0, -1).trim() : name.trim();
};

export default function PredictSection() {
  const [ref, isVisible] = useIntersectionObserver<HTMLElement>();
  
  // State variables for datasets and engine
  const [players, setPlayers] = useState<Player[]>([]);
  const [modelConfig, setModelConfig] = useState<ModelConfig | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [selectedPlayer, setSelectedPlayer] = useState<Player | null>(null);
  
  // Search query and dropdown state
  const [searchQuery, setSearchQuery] = useState<string>('');
  const [dropdownOpen, setDropdownOpen] = useState<boolean>(false);
  
  // Slider statistics values state
  const [sliderValues, setSliderValues] = useState<Record<string, number>>({});
  
  // Headshot image loading error handling state
  const [imageError, setImageError] = useState<boolean>(false);
  
  const searchContainerRef = useRef<HTMLDivElement>(null);

  // A. Load static JSON data assets on mount
  useEffect(() => {
    const loadAssets = async () => {
      try {
        const [playersRes, configRes] = await Promise.all([
          fetch('/players_data.json'),
          fetch('/model_config.json')
        ]);
        const playersData: Player[] = await playersRes.json();
        const configData: ModelConfig = await configRes.json();

        setPlayers(playersData);
        setModelConfig(configData);

        // Select Michael Jordan by default, or fallback to the first record
        const defaultPlayer = playersData.find(p => cleanPlayerName(p.name) === "Michael Jordan") || playersData[0];
        if (defaultPlayer) {
          handleSelectPlayer(defaultPlayer, configData);
        }
        setIsLoading(false);
      } catch (error) {
        console.error("Failed to load NBA machine learning prediction assets:", error);
        setIsLoading(false);
      }
    };
    loadAssets();
  }, []);

  // B. Handle clicks outside the autocomplete search box to close dropdown
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (searchContainerRef.current && !searchContainerRef.current.contains(event.target as Node)) {
        setDropdownOpen(false);
      }
    };
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  // C. Reset image fallback whenever player changes
  useEffect(() => {
    setImageError(false);
  }, [selectedPlayer]);

  // D. Select a player and populate their rookie statistics to the range sliders
  const handleSelectPlayer = (player: Player, config?: ModelConfig | null) => {
    const activeConfig = config || modelConfig;
    if (!activeConfig) return;
    setSelectedPlayer(player);
    setSearchQuery(cleanPlayerName(player.name));
    
    // Map stats or fallback to features imputer median
    const initialStats: Record<string, number> = {};
    activeConfig.features.forEach((feat, idx) => {
      const val = player.stats[feat];
      const median = activeConfig.imputer.medians[idx];
      initialStats[feat] = val !== null && val !== undefined ? val : median;
    });
    setSliderValues(initialStats);
  };

  // E. Handle slider modification
  const handleSliderChange = (feat: string, val: number) => {
    setSliderValues(prev => ({
      ...prev,
      [feat]: val
    }));
  };

  // F. Filter suggestions dynamically
  const filteredPlayers = useMemo(() => {
    if (!searchQuery) return [];
    const query = searchQuery.toLowerCase().trim();
    return players.filter(p => 
      cleanPlayerName(p.name).toLowerCase().includes(query)
    ).slice(0, 10);
  }, [searchQuery, players]);

  // G. Live machine learning predictions calculation
  const predictions = useMemo(() => {
    if (!modelConfig || Object.keys(sliderValues).length === 0) return null;
    const config = modelConfig;

    // 1. Order stats according to config features order
    const orderedStats = config.features.map(feat => sliderValues[feat] ?? 0);

    // 2. Perform Z-Score scaling: z = (x - mean) / scale
    const scaledStats = orderedStats.map((val, idx) => {
      const mean = config.scaler.mean[idx];
      const scale = config.scaler.scale[idx];
      return (val - mean) / scale;
    });

    // 3. Logistic Regression probability computation
    const lrWeights = config.logistic_regression.coefficients;
    const lrIntercept = config.logistic_regression.intercept;
    let lrZ = lrIntercept;
    for (let i = 0; i < scaledStats.length; i++) {
      lrZ += scaledStats[i] * lrWeights[i];
    }
    const lrProb = 1.0 / (1.0 + Math.exp(-lrZ));
    const lrPred = lrProb >= 0.5 ? 1 : 0;

    // 4. Decision Tree recursive traversal computation
    const traverseTree = (node: DecisionTreeNode, stats: number[]): { probability: number; prediction: number } => {
      if (node.leaf) {
        return {
          probability: node.probabilities ? node.probabilities[1] : 0.5,
          prediction: node.prediction ?? 0
        };
      }
      const featIdx = node.feature_idx!;
      const threshold = node.threshold!;
      const featureVal = stats[featIdx];

      if (featureVal <= threshold) {
        return traverseTree(node.left!, stats);
      } else {
        return traverseTree(node.right!, stats);
      }
    };

    const dtResult = traverseTree(config.decision_tree, scaledStats);
    const dtProb = dtResult.probability;
    const dtPred = dtResult.prediction;

    return {
      lrProb,
      lrPred,
      dtProb,
      dtPred
    };
  }, [sliderValues, modelConfig]);

  // H. Prepare UI data from prediction calculation
  const uiPredictions = useMemo(() => {
    if (!predictions) return null;

    const { lrProb, lrPred, dtProb, dtPred } = predictions;

    const lrConfidencePct = lrPred === 1 ? (lrProb * 100) : ((1 - lrProb) * 100);
    const dtConfidencePct = dtPred === 1 ? (dtProb * 100) : ((1 - dtProb) * 100);

    let consensusLabel = "N/A";
    let consensusColor = "bg-slate-700/50 border border-slate-600/50 text-slate-300";
    
    if (lrPred === dtPred) {
      if (lrPred === 1) {
        consensusLabel = "AGREE - YES (Sustainable)";
        consensusColor = "bg-emerald-500/20 border border-emerald-500/30 text-emerald-400";
      } else {
        consensusLabel = "AGREE - NO (< 5 Yrs)";
        consensusColor = "bg-rose-500/20 border border-rose-500/30 text-rose-400";
      }
    } else {
      consensusLabel = "DISAGREE (Mixed Predictions)";
      consensusColor = "bg-amber-500/20 border border-amber-500/30 text-amber-400";
    }

    return {
      lrOutcome: lrPred === 1 ? "YES (Sustainable)" : "NO (Under 5 Yrs)",
      lrConfidence: `${lrConfidencePct.toFixed(1)}%`,
      lrBarWidth: `${lrConfidencePct}%`,
      lrColor: lrPred === 1 ? "bg-emerald-500 shadow-[0_0_8px_rgba(16,185,129,0.5)]" : "bg-rose-500 shadow-[0_0_8px_rgba(239,68,68,0.5)]",
      lrTextClass: lrPred === 1 ? "text-emerald-400" : "text-rose-400",

      dtOutcome: dtPred === 1 ? "YES (Sustainable)" : "NO (Under 5 Yrs)",
      dtConfidence: `${dtConfidencePct.toFixed(1)}%`,
      dtBarWidth: `${dtConfidencePct}%`,
      dtColor: dtPred === 1 ? "bg-emerald-500 shadow-[0_0_8px_rgba(16,185,129,0.5)]" : "bg-rose-500 shadow-[0_0_8px_rgba(239,68,68,0.5)]",
      dtTextClass: dtPred === 1 ? "text-emerald-400" : "text-rose-400",

      consensusLabel,
      consensusColor
    };
  }, [predictions]);

  // Loading indicator rendering
  if (isLoading) {
    return (
      <section id="predict" className="py-24 px-6 bg-secondary text-primary-foreground w-full relative overflow-hidden flex items-center justify-center min-h-[600px]">
        <div className="absolute top-0 w-full h-2 bg-primary"></div>
        <div className="text-center flex flex-col items-center">
          <div className="w-16 h-16 border-4 border-primary border-t-transparent rounded-full animate-spin mb-6"></div>
          <h2 className="text-2xl font-display uppercase tracking-widest text-white/90">Loading ML Engine Assets...</h2>
          <p className="text-sm font-body text-blue-100/60 mt-2">Preparing historical datasets & ML parameters</p>
        </div>
      </section>
    );
  }

  const cleanName = selectedPlayer ? cleanPlayerName(selectedPlayer.name) : "";
  const headshotUrl = selectedPlayer && selectedPlayer.id 
    ? `https://ak-static.cms.nba.com/wp-content/uploads/headshots/nba/latest/260x190/${selectedPlayer.id}.png`
    : null;

  return (
    <section id="predict" ref={ref} className="py-24 px-6 bg-secondary text-primary-foreground w-full relative overflow-hidden">
      <div className="absolute top-0 w-full h-2 bg-primary"></div>
      <div className={`max-w-7xl mx-auto relative z-10 ${isVisible ? 'animate-fade-rise' : 'opacity-0'}`}>
        
        <div className="text-center flex flex-col items-center mb-16">
          <h2 className="text-5xl md:text-6xl font-display uppercase leading-none mb-6">Career Longevity Predictor</h2>
          <p className="text-xl max-w-3xl text-blue-100 font-body">
            Use Machine Learning to predict if a rookie will play 5+ years in the NBA based on their rookie season statistics.
          </p>
        </div>

        <div className="flex flex-col lg:flex-row gap-8 w-full">
          {/* Left Panel */}
          <div className={`flex-1 bg-white text-foreground p-8 shadow-xl border-t-8 border-primary flex flex-col ${isVisible ? 'animate-fade-rise-delay' : 'opacity-0'}`}>
            <h3 className="text-3xl font-display text-secondary uppercase mb-6 border-b-2 border-muted pb-4">1. Select & Modify Stats</h3>
            
            {/* Search Autocomplete bar */}
            <div className="mb-8 relative" ref={searchContainerRef}>
              <label className="block text-sm font-bold uppercase tracking-widest text-muted-foreground mb-2">Search NBA Player</label>
              <div className="relative">
                <Search className="absolute left-3 top-1/2 -translate-y-1/2 text-muted-foreground w-5 h-5" />
                <input 
                  type="text" 
                  value={searchQuery}
                  onChange={(e) => {
                    setSearchQuery(e.target.value);
                    setDropdownOpen(true);
                  }}
                  onFocus={() => setDropdownOpen(true)}
                  placeholder="e.g. Michael Jordan, Brandon Ingram..." 
                  className="w-full pl-10 pr-4 py-3 bg-muted border-2 border-transparent focus:border-primary outline-none transition-colors font-body font-medium"
                />
              </div>
              
              {/* Autocomplete Dropdown list */}
              {dropdownOpen && searchQuery.trim() !== '' && (
                <div className="absolute top-full left-0 right-0 bg-white border border-border mt-1 shadow-2xl rounded-none max-h-60 overflow-y-auto z-50 divide-y divide-muted custom-scrollbar">
                  {filteredPlayers.length > 0 ? (
                    filteredPlayers.map((player, idx) => {
                      const cName = cleanPlayerName(player.name);
                      const badgeLabel = player.actual_5yrs === 1 ? ">= 5 Yrs" : "< 5 Yrs";
                      const badgeClass = player.actual_5yrs === 1 
                        ? "bg-emerald-100 text-emerald-800 border border-emerald-200" 
                        : "bg-rose-100 text-rose-800 border border-rose-200";
                      
                      return (
                        <div 
                          key={idx} 
                          onClick={() => handleSelectPlayer(player)}
                          className="flex justify-between items-center px-4 py-3 hover:bg-muted cursor-pointer transition-colors duration-150"
                        >
                          <span className="font-body font-semibold text-secondary">{cName}</span>
                          <span className={`text-[10px] font-bold uppercase tracking-wider px-2 py-0.5 rounded ${badgeClass}`}>
                            {badgeLabel}
                          </span>
                        </div>
                      );
                    })
                  ) : (
                    <div className="px-4 py-3 text-sm text-muted-foreground italic text-center font-body">
                      No players found matching "{searchQuery}"
                    </div>
                  )}
                </div>
              )}
            </div>

            {/* Statistics display container */}
            <div className="flex-1 pr-4">
              <div className="mb-4">
                <h4 className="text-xl font-bold font-display text-secondary uppercase">Rookie Season Statistics</h4>
                <p className="text-sm text-muted-foreground mt-1 leading-relaxed">
                  Actual rookie season metrics for the selected NBA player used by the ML engine to evaluate career longevity.
                </p>
              </div>

              {modelConfig && (
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 mt-6 pb-6">
                  {modelConfig.features.map((feat) => {
                    const meta = FEATURE_META[feat] || { label: feat, min: 0, max: 100, step: 0.1 };
                    const value = sliderValues[feat] ?? 0;
                    const pct = Math.min(100, Math.max(0, ((value - meta.min) / (meta.max - meta.min)) * 100));
                    return (
                      <div key={feat} className="bg-muted/40 p-4 border border-muted flex flex-col justify-between gap-2 shadow-sm hover:shadow-md transition-shadow duration-200">
                        <div className="flex justify-between items-start">
                          <span className="text-xs font-bold uppercase tracking-wider text-secondary leading-tight max-w-[70%]">{meta.label}</span>
                          <span className="font-mono text-primary font-extrabold bg-white px-2 py-0.5 border border-muted text-sm rounded shadow-sm">
                            {value.toFixed(meta.step % 1 === 0 ? 0 : 1)}
                          </span>
                        </div>
                        <div className="w-full bg-slate-200/60 h-2 rounded-full overflow-hidden mt-1">
                          <div 
                            className="bg-secondary h-full transition-all duration-500 rounded-full"
                            style={{ width: `${pct}%` }}
                          ></div>
                        </div>
                      </div>
                    );
                  })}
                </div>
              )}
            </div>
          </div>

          {/* Right Panel */}
          <div className="flex-1 flex flex-col gap-8">
            
            {/* Player Profile Card */}
            <div className={`bg-white text-foreground p-8 shadow-xl border-t-8 border-secondary flex flex-col ${isVisible ? 'animate-fade-rise-delay-2' : 'opacity-0'}`}>
              <div className="flex gap-6 items-center">
                <div className="w-32 h-32 bg-muted rounded-full border-4 border-primary overflow-hidden flex-shrink-0 flex justify-center items-center relative shadow-inner">
                  {(!headshotUrl || imageError) ? (
                    <svg className="w-16 h-16 text-muted-foreground" fill="currentColor" viewBox="0 0 24 24">
                      <path d="M24 20.993V24H0v-2.996A14.977 14.977 0 0112.004 15c4.904 0 9.26 2.354 11.996 5.993zM16.002 8.999a4 4 0 11-8 0 4 4 0 018 0z" />
                    </svg>
                  ) : (
                    <img 
                      src={headshotUrl} 
                      alt={`${cleanName} Headshot`} 
                      className="w-full h-full object-cover transform hover:scale-105 transition-transform duration-300"
                      onError={() => setImageError(true)}
                    />
                  )}
                </div>
                <div>
                  <h2 className="text-4xl font-display text-secondary uppercase mb-2">
                    {selectedPlayer ? cleanName : "Select a Player"}
                  </h2>
                  <div className="flex items-center gap-3">
                    <span className="text-sm font-bold uppercase tracking-widest text-muted-foreground">Actual Career:</span>
                    {selectedPlayer ? (
                      <span className={`font-bold px-3 py-1 text-sm rounded ${
                        selectedPlayer.actual_5yrs === 1 
                          ? "bg-emerald-100 text-emerald-800 border border-emerald-200" 
                          : "bg-rose-100 text-rose-800 border border-rose-200"
                      }`}>
                        {selectedPlayer.actual_5yrs === 1 ? ">= 5 YEARS (Sustainable)" : "< 5 YEARS (Under 5 Yrs)"}
                      </span>
                    ) : (
                      <span className="bg-muted text-secondary font-bold px-3 py-1 text-sm rounded">N/A</span>
                    )}
                  </div>
                </div>
              </div>
            </div>

            {/* Model Predictions Output Cards */}
            <div className={`bg-secondary text-primary-foreground p-8 shadow-xl border-t-8 border-primary flex flex-col relative overflow-hidden ${isVisible ? 'animate-fade-rise-delay-2' : 'opacity-0'}`}>
              <h3 className="text-3xl font-display uppercase mb-8 pb-4 border-b border-primary/30 relative z-10">2. Machine Learning Predictions</h3>
              
              <div className="grid md:grid-cols-2 gap-8 relative z-10">
                {/* Logistic Regression Card */}
                <div className="bg-slate-900/40 p-6 border-2 border-primary/40 relative group hover:border-primary transition-all duration-300 shadow-md">
                  <h4 className="text-lg font-bold uppercase tracking-widest text-blue-200 mb-2">Logistic Regression</h4>
                  <div className={`text-3xl font-display uppercase mb-6 ${uiPredictions?.lrTextClass || "text-white"}`}>
                    {uiPredictions ? uiPredictions.lrOutcome : "--"}
                  </div>
                  
                  <div className="confidence-bar-wrapper">
                    <div className="flex justify-between items-center mb-1 font-mono text-xs">
                      <span className="text-blue-200">Confidence</span>
                      <span className="text-white font-bold">{uiPredictions ? uiPredictions.lrConfidence : "0%"}</span>
                    </div>
                    <div className="w-full bg-slate-950/70 h-2 rounded-full overflow-hidden border border-slate-800">
                      <div 
                        className={`h-full transition-all duration-500 ${uiPredictions?.lrColor || "bg-primary"}`}
                        style={{ width: uiPredictions ? uiPredictions.lrBarWidth : "0%" }}
                      ></div>
                    </div>
                  </div>
                </div>

                {/* Decision Tree Card */}
                <div className="bg-slate-900/40 p-6 border-2 border-primary/40 relative group hover:border-primary transition-all duration-300 shadow-md">
                  <h4 className="text-lg font-bold uppercase tracking-widest text-blue-200 mb-2">Decision Tree</h4>
                  <div className={`text-3xl font-display uppercase mb-6 ${uiPredictions?.dtTextClass || "text-white"}`}>
                    {uiPredictions ? uiPredictions.dtOutcome : "--"}
                  </div>
                  
                  <div className="confidence-bar-wrapper">
                    <div className="flex justify-between items-center mb-1 font-mono text-xs">
                      <span className="text-blue-200">Confidence</span>
                      <span className="text-white font-bold">{uiPredictions ? uiPredictions.dtConfidence : "0%"}</span>
                    </div>
                    <div className="w-full bg-slate-950/70 h-2 rounded-full overflow-hidden border border-slate-800">
                      <div 
                        className={`h-full transition-all duration-500 ${uiPredictions?.dtColor || "bg-primary"}`}
                        style={{ width: uiPredictions ? uiPredictions.dtBarWidth : "0%" }}
                      ></div>
                    </div>
                  </div>
                </div>
              </div>
              
              {/* Model Consensus Banner */}
              <div className={`mt-8 p-5 flex justify-center items-center gap-4 relative z-10 transition-all duration-300 ${
                uiPredictions ? uiPredictions.consensusColor : "bg-primary text-white"
              }`}>
                 <span className="font-bold uppercase tracking-widest text-xs opacity-90">Model Consensus:</span>
                 <span className="text-2xl font-display uppercase tracking-wider">
                   {uiPredictions ? uiPredictions.consensusLabel : "N/A"}
                 </span>
              </div>
            </div>
          </div>

        </div>
      </div>
    </section>
  );
}
