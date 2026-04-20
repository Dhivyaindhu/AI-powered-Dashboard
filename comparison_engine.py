"""
==============================================================
 comparison_engine.py
 The CORRECT way to compare Gold vs Stocks vs Crypto
==============================================================

THE PROBLEM WITH THE OLD CODE:
  It compared raw prices — Gold ₹7,000 vs BTC ₹60,00,000.
  These numbers mean nothing side-by-side.

THE FIX — Budget-Anchored Comparison:
  Step 1: Take user budget (e.g. ₹50,000)
  Step 2: Calculate exactly how many units of each asset they can buy
  Step 3: Project each to the target date using forecasts
  Step 4: Compare FINAL VALUE and % GROWTH — apples to apples

  Example with ₹50,000:
  • Gold     → buys 6.5 grams  → worth ₹56,000 by June 2027 → +12%
  • NIFTY    → buys 2.1 units  → worth ₹59,000 by June 2027 → +18%
  • BTC      → buys 0.0008 BTC → worth ₹72,000 by June 2027 → +44%

  Winner is obvious now. The unit size doesn't matter.

WHAT THIS FILE PROVIDES:
  - BudgetComparison class: core comparison logic
  - compute_budget_scenarios(): multi-scenario analysis
  - get_recommendation(): final verdict with reasoning
==============================================================
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
import warnings

warnings.filterwarnings("ignore")


# ──────────────────────────────────────────────
#  ASSET TYPE DEFINITIONS
#  Different assets need different treatment for
#  "how many units does my budget buy?"
# ──────────────────────────────────────────────

ASSET_TYPES = {
    # Asset name → (type, min_unit, display_unit)
    # type: "gram" | "share" | "crypto" | "index"
    "Gold"    : ("gram",   1.0,    "grams"),
    "Stocks"  : ("index",  1.0,    "units"),   # NIFTY/S&P index units
    "Bitcoin" : ("crypto", 0.0001, "BTC"),     # fractional BTC allowed
    "Ethereum": ("crypto", 0.001,  "ETH"),     # fractional ETH allowed
    "NIFTY"   : ("index",  1.0,    "units"),
    "S&P 500" : ("index",  1.0,    "units"),
    "Sensex"  : ("index",  1.0,    "units"),
}

# Minimum investable amount (exchange/platform limits)
MIN_INVESTMENT = {
    "gram"  : 1,     # 1 gram minimum for digital gold
    "index" : 100,   # ₹100 via index funds / ETF
    "crypto": 100,   # ₹100 minimum on most exchanges
    "share" : 1,     # 1 share minimum for individual stocks
}


# ──────────────────────────────────────────────
#  CORE DATA STRUCTURES
# ──────────────────────────────────────────────

@dataclass
class AssetSnapshot:
    """Current state + forecast for one asset."""
    name            : str
    current_price   : float         # price per unit in local currency
    forecast_prices : pd.Series     # date-indexed forecast series
    unit            : str           # "gram", "BTC", "point", etc.
    asset_type      : str           # "gram" | "crypto" | "index"
    currency        : str           # "₹" or "$"
    historical_df   : pd.DataFrame  # full cleaned history with features

    # Computed automatically in __post_init__
    daily_returns  : pd.Series = field(default_factory=pd.Series)
    volatility_30d : float = 0.0
    rsi            : float = 50.0
    trend          : str = "Neutral"
    max_drawdown   : float = 0.0

    def __post_init__(self):
        if not self.historical_df.empty and "Close" in self.historical_df.columns:
            close = self.historical_df["Close"].dropna()
            self.daily_returns  = close.pct_change().dropna()
            self.volatility_30d = (
                float(self.daily_returns.rolling(30).std().iloc[-1])
                if len(self.daily_returns) >= 30 else 0.0
            )
            self.rsi = (
                float(self.historical_df["RSI"].dropna().iloc[-1])
                if "RSI" in self.historical_df.columns and not self.historical_df["RSI"].dropna().empty
                else 50.0
            )
            self.max_drawdown = self._compute_max_drawdown(close)
            self.trend        = self._get_trend()

    def _compute_max_drawdown(self, prices: pd.Series) -> float:
        """Max drawdown over the last 2 years."""
        try:
            recent = prices.last("730D") if len(prices) > 0 else prices
        except Exception:
            recent = prices.tail(730)
        if recent.empty:
            return 0.0
        roll_max = recent.cummax()
        drawdown = (recent - roll_max) / (roll_max + 1e-9)
        return float(drawdown.min())

    def _get_trend(self) -> str:
        """Simple trend: Bullish / Bearish / Neutral based on MA crossover."""
        try:
            if (
                "ma_200d" in self.historical_df.columns
                and "ma_30d" in self.historical_df.columns
            ):
                ma200 = float(self.historical_df["ma_200d"].dropna().iloc[-1])
                ma30  = float(self.historical_df["ma_30d"].dropna().iloc[-1])
                if self.current_price > ma200 and ma30 > ma200:
                    return "Bullish"
                elif self.current_price < ma200 and ma30 < ma200:
                    return "Bearish"
        except Exception:
            pass
        return "Neutral"

    def price_at(self, target_date: pd.Timestamp) -> Optional[float]:
        """Get forecast price at or nearest to the target date."""
        if self.forecast_prices is None or self.forecast_prices.empty:
            return None
        try:
            idx = self.forecast_prices.index.get_indexer([target_date], method="nearest")[0]
            if idx < 0:
                return None
            return float(self.forecast_prices.iloc[idx])
        except Exception:
            return None


@dataclass
class AssetResult:
    """Full comparison result for one asset."""
    name                 : str
    current_price        : float
    forecast_price       : float
    units_bought         : float      # how many units budget buys at current price
    value_at_target      : float      # units_bought × forecast_price
    absolute_profit      : float      # value_at_target - budget
    return_pct           : float      # (value_at_target - budget) / budget * 100
    annualised_return_pct: float
    volatility           : float      # 30-day daily std
    risk_adjusted_return : float      # Sharpe-like: return_pct / (ann_vol% + 1)
    max_drawdown         : float
    rsi                  : float
    trend                : str
    recommendation       : str        # e.g. "✅ Strong Buy", "🟢 Buy", "⚪ Watch"
    reasoning            : str        # human-readable explanation
    score                : float      # 0–100 composite score
    currency             : str
    unit                 : str


# ──────────────────────────────────────────────
#  COMPARISON ENGINE
# ──────────────────────────────────────────────

class BudgetComparison:
    """
    Core class: given a budget and target date, which asset gives the
    best risk-adjusted outcome?

    Usage:
        engine = BudgetComparison(budget=50000, target_date="2027-06-01",
                                  currency="₹", risk_profile="moderate")
        engine.add_asset(snapshot)   # one AssetSnapshot per asset
        results = engine.run()       # returns list[AssetResult], sorted best→worst
        reco    = engine.recommend() # returns full recommendation dict
    """

    # Risk-profile scoring weights
    RISK_WEIGHTS = {
        "conservative": {"return": 0.25, "volatility": 0.40, "trend": 0.20, "affordability": 0.15},
        "moderate"    : {"return": 0.40, "volatility": 0.25, "trend": 0.20, "affordability": 0.15},
        "aggressive"  : {"return": 0.60, "volatility": 0.10, "trend": 0.20, "affordability": 0.10},
    }

    def __init__(
        self,
        budget      : float,
        target_date : str,
        currency    : str = "₹",
        risk_profile: str = "moderate",
    ):
        self.budget       = float(budget)
        self.target_date  = pd.Timestamp(target_date)
        self.currency     = currency
        self.risk_profile = risk_profile.lower().strip()
        self.assets  : list[AssetSnapshot] = []
        self.results : list[AssetResult]   = []
        self.today        = pd.Timestamp.today().normalize()
        self.days_ahead   = max(1, (self.target_date - self.today).days)
        self.years_ahead  = self.days_ahead / 365.25

    def add_asset(self, snapshot: AssetSnapshot):
        """Register an asset for comparison."""
        self.assets.append(snapshot)

    # ── Step 1: How many units does the budget buy? ──────────
    def _units_bought(self, snapshot: AssetSnapshot) -> float:
        """
        Crypto & digital gold: fractional units allowed.
        Direct stock/index purchase: floor to whole units.
        (If investing via ETF/index fund, fractional is fine — this is
         the conservative market-behaviour estimate.)
        """
        if snapshot.current_price <= 0:
            return 0.0
        units = self.budget / snapshot.current_price
        if snapshot.asset_type in ("index", "share"):
            units = np.floor(units)
        return float(units)

    # ── Step 2: What is those units worth at target date? ───
    def _value_at_target(self, units: float, snapshot: AssetSnapshot) -> float:
        future_price = snapshot.price_at(self.target_date)
        if future_price is None or future_price <= 0:
            return units * snapshot.current_price   # flat forecast fallback
        return units * future_price

    # ── Step 3: Score the asset (0–100) ─────────────────────
    def _score(self, result: AssetResult) -> float:
        """
        Composite score weighted by the user's risk profile.
        Higher = better investment for this user.
        """
        weights = self.RISK_WEIGHTS.get(
            self.risk_profile, self.RISK_WEIGHTS["moderate"]
        )

        # Return score (0–1): maps -50% → 0,  +100% → 1
        ret_norm = float(np.clip((result.return_pct + 50) / 150, 0, 1))

        # Volatility score (0–1): lower annualised vol = better
        ann_vol_pct = result.volatility * np.sqrt(252) * 100
        vol_score   = float(np.clip(1 - ann_vol_pct / 100, 0, 1))

        # Trend score
        trend_map  = {"Bullish": 1.0, "Neutral": 0.6, "Bearish": 0.2}
        trend_score = trend_map.get(result.trend, 0.6)

        # Affordability: log-scaled so BTC (0.0008 units) isn't penalised
        # unfairly vs Gold (6 grams)
        afford = float(
            np.clip(np.log10(result.units_bought + 0.001) / 3 + 0.5, 0, 1)
            if result.units_bought > 0 else 0.0
        )

        composite = (
            weights["return"]        * ret_norm    +
            weights["volatility"]    * vol_score   +
            weights["trend"]         * trend_score +
            weights["affordability"] * afford
        ) * 100

        return round(float(composite), 2)

    # ── Step 4: Generate human-readable recommendation ───────
    def _generate_recommendation(self, result: AssetResult) -> tuple[str, str]:
        """Return (signal_tag, reasoning_text) for the asset."""
        ret  = result.return_pct
        vol  = result.volatility * np.sqrt(252) * 100   # annualised %
        rsi  = result.rsi
        risk = self.risk_profile

        reasons = []

        # Expected return
        if ret > 20:
            reasons.append(f"strong expected growth of {ret:.1f}%")
        elif ret > 10:
            reasons.append(f"solid expected return of {ret:.1f}%")
        elif ret > 0:
            reasons.append(f"modest positive return of {ret:.1f}%")
        elif ret > -5:
            reasons.append(f"near-flat forecast ({ret:.1f}%)")
        else:
            reasons.append(f"negative forecast ({ret:.1f}%)")

        # Volatility
        if vol < 10:
            reasons.append("very low volatility")
        elif vol < 25:
            reasons.append("moderate volatility")
        elif vol < 60:
            reasons.append("high volatility")
        else:
            reasons.append("extremely high volatility — be careful")

        # RSI signal
        if rsi < 35:
            reasons.append("currently oversold (potential buying opportunity)")
        elif rsi > 65:
            reasons.append("currently overbought (may correct soon)")

        # Trend
        if result.trend == "Bullish":
            reasons.append("price above long-term trend")
        elif result.trend == "Bearish":
            reasons.append("price below long-term trend")

        # Max drawdown warning
        if result.max_drawdown < -0.30:
            reasons.append(f"significant historical drawdown of {result.max_drawdown * 100:.0f}%")

        reasoning = f"{result.name}: {', '.join(reasons)}."

        # Signal tag
        if ret > 15 and (vol < 50 or risk == "aggressive"):
            tag = "✅ Strong Buy"
        elif ret > 7:
            tag = "🟢 Buy"
        elif ret > 2:
            tag = "🟡 Hold / Accumulate"
        elif ret > -5:
            tag = "⚪ Watch"
        else:
            tag = "🔴 Avoid"

        return tag, reasoning

    # ── Main run ─────────────────────────────────────────────
    def run(self) -> list[AssetResult]:
        """
        Run the full budget-anchored comparison.
        Returns list[AssetResult] sorted best → worst by score.
        """
        self.results = []

        for snapshot in self.assets:
            if snapshot.current_price <= 0:
                continue

            units      = self._units_bought(snapshot)
            future_val = self._value_at_target(units, snapshot)
            profit     = future_val - self.budget
            ret_pct    = (profit / self.budget) * 100 if self.budget > 0 else 0.0
            ann_ret    = (
                ((1 + ret_pct / 100) ** (1 / max(self.years_ahead, 0.1)) - 1) * 100
            )
            fp = snapshot.price_at(self.target_date) or snapshot.current_price
            ann_vol = snapshot.volatility_30d * np.sqrt(252) * 100

            result = AssetResult(
                name                  = snapshot.name,
                current_price         = snapshot.current_price,
                forecast_price        = fp,
                units_bought          = units,
                value_at_target       = future_val,
                absolute_profit       = profit,
                return_pct            = ret_pct,
                annualised_return_pct = ann_ret,
                volatility            = snapshot.volatility_30d,
                risk_adjusted_return  = ret_pct / (ann_vol + 1),
                max_drawdown          = snapshot.max_drawdown,
                rsi                   = snapshot.rsi,
                trend                 = snapshot.trend,
                recommendation        = "",   # filled below
                reasoning             = "",
                score                 = 0.0,
                currency              = self.currency,
                unit                  = snapshot.unit,
            )

            result.score                        = self._score(result)
            result.recommendation, result.reasoning = self._generate_recommendation(result)

            self.results.append(result)

        self.results.sort(key=lambda r: r.score, reverse=True)
        return self.results

    # ── Final recommendation dict ─────────────────────────────
    def recommend(self) -> dict:
        """
        Top-level recommendation with explanation and split strategy.
        Must call run() first.
        """
        if not self.results:
            return {"error": "No results. Call run() first."}

        best  = self.results[0]
        worst = self.results[-1]

        # Overall verdict
        if best.return_pct > 20:
            verdict    = "🟢 Clear Winner Found"
            confidence = "High"
        elif best.return_pct > 8:
            verdict    = "🟡 Moderate Opportunity"
            confidence = "Medium"
        elif best.return_pct > 0:
            verdict    = "⚪ Marginal Gains Expected"
            confidence = "Low"
        else:
            verdict    = "🔴 All Assets Look Weak Right Now"
            confidence = "Low"

        # Human-readable explanation block
        explanation_lines = [
            f"For a budget of {self.currency}{self.budget:,.0f} invested today "
            f"until {self.target_date.strftime('%B %Y')} ({self.days_ahead} days):",
        ]
        for r in self.results:
            explanation_lines.append(
                f"  • {r.name}: Buy {r.units_bought:.4f} {r.unit} → "
                f"worth {self.currency}{r.value_at_target:,.0f} "
                f"({r.return_pct:+.1f}%)  [Score: {r.score:.0f}/100]"
            )
        explanation_lines.append(
            f"\n→ Best choice for your {self.risk_profile} risk profile: {best.name}"
        )

        # Split strategy: suggest 60/40 if #2 asset is also strong
        positive_assets  = [r for r in self.results if r.return_pct > 0]
        split_suggestion = None
        if len(positive_assets) >= 2:
            top2 = positive_assets[:2]
            if top2[1].return_pct > top2[0].return_pct * 0.6:
                split_suggestion = {
                    "asset1":         top2[0].name,
                    "pct1":           60,
                    "asset2":         top2[1].name,
                    "pct2":           40,
                    "blended_return": top2[0].return_pct * 0.6 + top2[1].return_pct * 0.4,
                }

        return {
            "verdict"         : verdict,
            "confidence"      : confidence,
            "best_asset"      : best.name,
            "best_score"      : best.score,
            "best_return_pct" : best.return_pct,
            "best_value"      : best.value_at_target,
            "best_profit"     : best.absolute_profit,
            "explanation"     : "\n".join(explanation_lines),
            "results"         : self.results,
            "split_suggestion": split_suggestion,
            "budget"          : self.budget,
            "currency"        : self.currency,
            "target_date"     : self.target_date,
            "days_ahead"      : self.days_ahead,
            "risk_profile"    : self.risk_profile,
        }


# ──────────────────────────────────────────────
#  SCENARIO ANALYSIS: What if my budget changes?
# ──────────────────────────────────────────────

def compute_budget_scenarios(
    asset_snapshots : list[AssetSnapshot],
    target_date     : str,
    budgets         : list[float],
    currency        : str = "₹",
    risk_profile    : str = "moderate",
) -> pd.DataFrame:
    """
    Run BudgetComparison for multiple budget sizes.
    Returns a tidy DataFrame — one row per (budget, asset) combination.

    Columns:
        budget, asset, value_at_target, return_pct,
        profit, score, units_bought
    """
    rows = []
    for budget in budgets:
        engine = BudgetComparison(budget, target_date, currency, risk_profile)
        for snap in asset_snapshots:
            engine.add_asset(snap)
        results = engine.run()
        for r in results:
            rows.append({
                "budget"          : budget,
                "asset"           : r.name,
                "value_at_target" : r.value_at_target,
                "return_pct"      : r.return_pct,
                "profit"          : r.absolute_profit,
                "score"           : r.score,
                "units_bought"    : r.units_bought,
            })
    return pd.DataFrame(rows)
