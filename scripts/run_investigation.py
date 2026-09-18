#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
from collections.abc import Callable

import pandas as pd

from config.settings import Config
from src.analyzers.anomaly_detector import AnomalyDetector
from src.analyzers.benford_law import BenfordLawAnalyzer
from src.analyzers.concentration import ConcentrationAnalyzer
from src.analyzers.cost_basis_analyzer import CostBasisAnalyzer
from src.analyzers.cross_fund_issuer import CrossFundIssuerAnalyzer
from src.analyzers.enhanced_phantom_assets import EnhancedPhantomAssetDetector
from src.analyzers.fraud_schemes import FraudSchemeDetector
from src.analyzers.fund_lifecycle import FundLifecycleAnalyzer
from src.analyzers.manager_network import ManagerNetworkAnalyzer
from src.analyzers.market_data import YFINANCE_AVAILABLE, MarketDataValidator
from src.analyzers.peer_comparison import PeerComparisonAnalyzer
from src.analyzers.portfolio_reconciliation import PortfolioReconciliationAnalyzer
from src.analyzers.price_divergence import CrossFundPriceDivergenceAnalyzer
from src.analyzers.quotaholder_analyzer import QuotaholderAnalyzer
from src.analyzers.valuation_smoothing import ValuationSmoothingAnalyzer
from src.analyzers.window_dressing import WindowDressingDetector
from src.enrichment.context_writer import build_context_section
from src.enrichment.exa_provider import ExaProvider
from src.enrichment.interfaces import SearchResult
from src.explain.explainer import Explainer
from src.processors.data_processor import DataProcessor
from src.utils.cnpj_utils import normalize_cnpj_list
from src.utils.fund_selector import FundSelector

logger = logging.getLogger(__name__)

_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")


def sanitize_run_id(run_id: str) -> str:
    """Return *run_id* if it matches the safe pattern, otherwise a timestamp fallback.

    Rejects absolute paths, path-traversal segments, and any characters that
    could affect file-system placement when used as a directory component.

    Parameters
    ----------
    run_id:
        The candidate run identifier supplied by the user.

    Returns
    -------
    str
        *run_id* unchanged when it matches ``^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$``;
        otherwise a ``%Y%m%d_%H%M%S`` timestamp string.
    """
    if _RUN_ID_RE.match(run_id):
        return run_id
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _normalize_selected_cnpjs(values: list[str] | None) -> list[str]:
    """Normalize user-supplied CNPJs, dropping duplicates and unusable values."""
    normalized = normalize_cnpj_list(values or [])
    return list(dict.fromkeys(normalized))


ANALYSIS_CHOICES = (
    "flow",
    "schemes",
    "phantom_assets",
    "quotaholder",
    "cost_basis",
    "reconciliation",
    "lifecycle",
    "manager_network",
    "window_dressing",
    "valuation_smoothing",
    "cross_fund_issuer",
    "price_divergence",
    "benford",
    "concentration",
    "peer_comparison",
    "market_data",
)


@dataclass(frozen=True)
class LoadedData:
    cadastro: pd.DataFrame | None
    informe: pd.DataFrame | None
    cda: pd.DataFrame | None
    #: CNPJs the run was scoped to, after resolving --fund-mode. Empty = no filter.
    selected_cnpjs: list[str] = field(default_factory=list)


def _has(df: pd.DataFrame | None, *required_columns: str) -> bool:
    """True when *df* holds rows and every required column."""
    if df is None or df.empty:
        return False
    return all(column in df.columns for column in required_columns)


@dataclass(frozen=True)
class AnalyzerSpec:
    """One optional analyzer: when it can run, and what it produces.

    Args:
        name: Selector used by --analysis and ANALYSIS_CHOICES
        is_runnable: Predicate over LoadedData deciding whether inputs suffice
        run: Callable returning {finding_key: DataFrame}
    """

    name: str
    is_runnable: Callable[[LoadedData], bool]
    run: Callable[[LoadedData], dict[str, pd.DataFrame]]


def _analyzer_specs(config: Config) -> tuple[AnalyzerSpec, ...]:
    """Declare the optional analyzers, their input requirements and their calls."""

    def _cda(loaded: LoadedData) -> pd.DataFrame:
        return _normalize_cda_columns(loaded.cda)

    return (
        AnalyzerSpec(
            name="quotaholder",
            is_runnable=lambda d: _has(d.informe, "NR_COTST"),
            run=lambda d: {
                "quotaholder_anomalies": QuotaholderAnalyzer(config=config).analyze(
                    d.informe, cadastro_df=d.cadastro
                )
            },
        ),
        AnalyzerSpec(
            name="cost_basis",
            is_runnable=lambda d: _has(d.cda, "VL_CUSTO_POS_FINAL"),
            run=lambda d: {
                "cost_basis_anomalies": CostBasisAnalyzer(config=config).analyze(_cda(d))
            },
        ),
        AnalyzerSpec(
            name="reconciliation",
            is_runnable=lambda d: _has(d.cda) and _has(d.informe),
            run=lambda d: {
                "reconciliation_gaps": PortfolioReconciliationAnalyzer(config=config).analyze(
                    _cda(d), d.informe
                )
            },
        ),
        AnalyzerSpec(
            name="lifecycle",
            is_runnable=lambda d: _has(d.cadastro, "DT_CONST"),
            run=lambda d: {
                "lifecycle_anomalies": FundLifecycleAnalyzer(config=config).analyze(
                    d.cadastro, informe_df=d.informe
                )
            },
        ),
        AnalyzerSpec(
            name="manager_network",
            is_runnable=lambda d: _has(d.cadastro, "CNPJ_GESTOR"),
            run=lambda d: {
                "manager_network_anomalies": ManagerNetworkAnalyzer(config=config).analyze(
                    d.cadastro,
                    cda_df=_cda(d) if _has(d.cda) else None,
                    informe_df=d.informe,
                )
            },
        ),
        AnalyzerSpec(
            name="window_dressing",
            is_runnable=lambda d: _has(d.informe, "VL_QUOTA"),
            run=lambda d: {
                "window_dressing": WindowDressingDetector(config=config).analyze(d.informe)
            },
        ),
        AnalyzerSpec(
            name="valuation_smoothing",
            is_runnable=lambda d: _has(d.informe, "VL_QUOTA"),
            run=lambda d: {
                "valuation_smoothing": ValuationSmoothingAnalyzer(config=config).analyze(
                    d.informe
                )
            },
        ),
        AnalyzerSpec(
            name="cross_fund_issuer",
            is_runnable=lambda d: _has(d.cda) and _has(d.cadastro),
            run=lambda d: {
                "cross_fund_issuer": CrossFundIssuerAnalyzer(config=config).analyze(
                    _cda(d), d.cadastro
                )
            },
        ),
        AnalyzerSpec(
            name="price_divergence",
            is_runnable=lambda d: _has(d.cda, "CD_ATIVO", "DT_COMPTC", "QT_POS"),
            run=lambda d: {
                "cross_fund_price_divergence": CrossFundPriceDivergenceAnalyzer(
                    config=config
                ).analyze(_cda(d))
            },
        ),
        AnalyzerSpec(
            name="benford",
            is_runnable=lambda d: _has(d.informe, "VL_PATRIM_LIQ"),
            run=lambda d: {
                "benford_violations": BenfordLawAnalyzer().analyze_fund_data(
                    d.informe, cda_df=_cda(d) if _has(d.cda) else None
                )
            },
        ),
        AnalyzerSpec(
            name="concentration",
            is_runnable=lambda d: _has(d.cda, "CD_ATIVO", "VL_MERCADO"),
            run=lambda d: {
                "concentration_violations": _run_concentration(d)
            },
        ),
        AnalyzerSpec(
            name="peer_comparison",
            is_runnable=lambda d: _has(d.informe, "VL_QUOTA") and _has(d.cadastro),
            run=lambda d: {
                "peer_outliers": _run_peer_comparison(d)
            },
        ),
        AnalyzerSpec(
            name="market_data",
            # Optional: needs yfinance, which is not a core dependency. Skips
            # rather than fails when it is absent, so a default install still
            # reports a complete run.
            is_runnable=lambda d: YFINANCE_AVAILABLE and _has(
                d.cda, "CD_ATIVO", "VL_MERCADO", "QT_POS", "DT_COMPTC"
            ),
            run=lambda d: {
                "price_manipulation": MarketDataValidator().detect_price_manipulation(
                    MarketDataValidator().validate_portfolio_prices(
                        _cda(d), sample_size=1000
                    )
                )
            },
        ),
    )


def _run_concentration(loaded: LoadedData) -> pd.DataFrame:
    """Flag excessive single-position concentration, by CVM fund category.

    Without a category every fund defaults to the 25% single-issuer limit,
    which is wrong for e.g. FIXED_INCOME (20%). Categories come from the
    registry's CLASSE column, same mapping PeerComparisonAnalyzer uses.
    """
    analyzer = ConcentrationAnalyzer()
    cadastro = loaded.cadastro
    if _has(cadastro, "CNPJ_FUNDO"):
        class_mapping = {
            "Fundo de Ações": "EQUITY",
            "Fundo de Renda Fixa": "FIXED_INCOME",
            "Fundo Multimercado": "MULTI_MARKET",
            "Fundo Cambial": "FX",
            "Fundo de Investimento Imobiliário": "REAL_ESTATE",
        }
        categories = cadastro["CLASSE"].fillna("UNKNOWN").map(class_mapping).fillna("DEFAULT") \
            if "CLASSE" in cadastro.columns else pd.Series("DEFAULT", index=cadastro.index)
        analyzer.load_fund_categories(
            dict(zip(cadastro["CNPJ_FUNDO"], categories, strict=True))
        )
    return analyzer.detect_excessive_concentration(_normalize_cda_columns(loaded.cda))


def _run_peer_comparison(loaded: LoadedData) -> pd.DataFrame:
    """Compare each in-scope fund's risk/return profile against its peers."""
    analyzer = PeerComparisonAnalyzer()
    analyzer.load_fund_categories(loaded.cadastro)

    metrics = analyzer.calculate_fund_metrics(loaded.informe)
    if metrics.empty:
        return pd.DataFrame()

    return analyzer.compare_with_peers(
        metrics["CNPJ_FUNDO"].dropna().astype(str).tolist(), metrics
    )


def run_investigation(args: argparse.Namespace) -> int:
    config = Config()
    selected_analyses = _selected_analyses(args.analysis)

    run_id = sanitize_run_id(args.run_id) if args.run_id else datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) if args.output_dir else (config.REPORTS_DIR / "investigation" / run_id)
    output_dir.mkdir(parents=True, exist_ok=True)

    processor = DataProcessor(config)
    loaded = _load_data(args=args, config=config, processor=processor)

    results: dict[str, pd.DataFrame] = {}
    sources: dict[str, str] = {}
    failures: list[str] = []
    skipped: list[str] = []

    findings_dir = output_dir / "findings"
    findings_dir.mkdir(parents=True, exist_ok=True)

    # --- Core detectors (public-data based) ---
    if "flow" in selected_analyses and loaded.informe is not None and not loaded.informe.empty:
        try:
            informe = loaded.informe.copy()
            informe = processor.calculate_net_flow(informe)

            anomaly_detector = AnomalyDetector(config=config)
            anomaly_report = anomaly_detector.generate_anomaly_report(informe, cda_df=loaded.cda)
            for key, df in anomaly_report.items():
                results[key] = df
                sources[key] = _save_finding(df, findings_dir / f"{key}.csv")
        except Exception:
            logger.exception("Analyzer flow failed")
            failures.append("flow")

    if (
        "schemes" in selected_analyses
        and loaded.cda is not None
        and not loaded.cda.empty
        and loaded.cadastro is not None
        and not loaded.cadastro.empty
        and loaded.informe is not None
        and not loaded.informe.empty
    ):
        try:
            cda = _normalize_cda_columns(loaded.cda)

            scheme_detector = FraudSchemeDetector()
            schemes = scheme_detector.generate_fraud_scheme_report(
                informe_df=loaded.informe,
                cda_df=cda,
                cadastro_df=loaded.cadastro,
                output_path=None,
            )
            # Normalize scheme keys to match explainer expectations.
            scheme_key_map = {
                "circular_flow": "circular_flow",
                "layered_funds": "layered_funds",
                "asset_inflation": "asset_inflation",
                "shell_networks": "shell_networks",
            }
            for raw_key, df in schemes.items():
                key = scheme_key_map.get(raw_key, raw_key)
                results[key] = df
                sources[key] = _save_finding(df, findings_dir / f"{key}.csv")
        except Exception:
            logger.exception("Analyzer schemes failed")
            failures.append("schemes")

    if "phantom_assets" in selected_analyses and loaded.cda is not None and not loaded.cda.empty:
        try:
            cda = _normalize_cda_columns(loaded.cda)
            phantom = EnhancedPhantomAssetDetector(cache_dir=Path("data/cache"))
            # Best-effort: load valid funds list when cadastro file exists on disk.
            if args.cadastro and Path(args.cadastro).exists():
                try:
                    phantom.load_funds_from_cadastro(Path(args.cadastro))
                except Exception:
                    logger.warning(
                        "Could not load cadastro fund list from %s; phantom-asset "
                        "detection will run without it", args.cadastro, exc_info=True
                    )
            phantom.update_registries()
            phantom_assets = phantom.detect_enhanced_phantom_assets(cda)
            results["phantom_assets"] = phantom_assets
            sources["phantom_assets"] = _save_finding(phantom_assets, findings_dir / "phantom_assets.csv")

            phantom_by_fund = _phantom_assets_by_fund(cda_df=cda, phantom_assets=phantom_assets)
            results["phantom_assets_by_fund"] = phantom_by_fund
            sources["phantom_assets_by_fund"] = _save_finding(
                phantom_by_fund, findings_dir / "phantom_assets_by_fund.csv"
            )
        except Exception:
            logger.exception("Analyzer phantom_assets failed")
            failures.append("phantom_assets")

    # --- New analyzers ---
    #
    # Each entry declares when it can run and how. Previously these were nine
    # near-identical blocks each ending in `except Exception: pass`, which made a
    # crashed analyzer indistinguishable from one that found nothing -- the worst
    # possible failure mode for a forensic tool.
    for spec in _analyzer_specs(config):
        if spec.name not in selected_analyses:
            continue
        if not spec.is_runnable(loaded):
            logger.debug("Skipping %s: required inputs not available", spec.name)
            skipped.append(spec.name)
            continue

        try:
            produced = spec.run(loaded)
        except Exception:
            logger.exception("Analyzer %s failed", spec.name)
            failures.append(spec.name)
            continue

        for key, df in produced.items():
            results[key] = df
            sources[key] = _save_finding(df, findings_dir / f"{key}.csv")

    if failures:
        logger.error(
            "%d of %d analyzers failed: %s. Findings below are incomplete.",
            len(failures),
            len(failures) + len(skipped) + len(results),
            ", ".join(failures),
        )

    _write_run_metadata(
        output_dir=output_dir,
        run_id=run_id,
        args=args,
        results=results,
        sources=sources,
        failures=failures,
        skipped=skipped,
        selected_cnpjs=loaded.selected_cnpjs,
        data_period=_data_period(loaded.informe),
    )

    exit_code = 1 if (failures and getattr(args, "strict", False)) else 0

    if not args.explain:
        return exit_code

    contexts_by_entity_id: dict[str, dict[str, Any]] = {}
    if args.enable_enrichment:
        contexts_by_entity_id = _run_context_enrichment(
            run_id=run_id,
            results=results,
            cadastro_df=loaded.cadastro,
            explain_max_entities=args.explain_max_entities,
            max_results=args.enrichment_max_results,
            since_days=args.enrichment_since_days,
            provider_name=args.enrichment_provider,
        )

    explainer = Explainer()
    explainer.generate(
        results=results,
        output_dir=output_dir,
        informe_df=loaded.informe,
        cadastro_df=loaded.cadastro,
        sources=sources,
        contexts_by_entity_id=contexts_by_entity_id,
        explain_max_entities=args.explain_max_entities,
        explain_top_findings=args.explain_top_findings,
        explain_format=args.explain_format,
    )

    return exit_code


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a REAG investigation and generate human-readable briefs.")
    parser.add_argument("--run-id", default=None, help="Optional run identifier (defaults to timestamp).")
    parser.add_argument("--output-dir", default=None, help="Output directory (defaults to reports/investigation/<run_id>).")

    parser.add_argument("--public-data-dir", default="data/raw", help="Base directory for public/raw data inputs.")
    parser.add_argument("--processed-data-dir", default="data/processed", help="Base directory for processed data inputs.")

    parser.add_argument("--cadastro", default=None, help="Path to cadastro CSV (cad_fi*.csv).")
    parser.add_argument("--informe", default=None, help="Path to informe CSV or processed file.")
    parser.add_argument("--cda", default=None, help="Path to CDA CSV.")

    parser.add_argument("--explain", action=argparse.BooleanOptionalAction, default=True, help="Generate per-entity briefs.")
    parser.add_argument("--explain-max-entities", type=int, default=200, help="Maximum number of entity briefs.")
    parser.add_argument("--explain-top-findings", type=int, default=6, help="Max evidence rows per brief.")
    parser.add_argument(
        "--explain-format",
        choices=["html", "md", "both"],
        default="both",
        help="Which formats to generate for briefs.",
    )
    parser.add_argument(
        "--analysis",
        nargs="+",
        choices=[*ANALYSIS_CHOICES, "all"],
        default=["all"],
        help="Which investigation modules to run. Defaults to all modules.",
    )
    parser.add_argument(
        "--fund-mode",
        choices=["administrator", "manager", "cnpj_list", "all"],
        default=None,
        help=(
            "Restrict the investigation to a subset of funds. 'administrator' and "
            "'manager' resolve --fund-identifier against the cadastro; 'cnpj_list' "
            "takes comma-separated CNPJs; 'all' (the default) applies no filter."
        ),
    )
    parser.add_argument(
        "--fund-identifier",
        default=None,
        help=(
            "Value for --fund-mode: an administrator or manager name (partial, "
            "case-insensitive), or comma-separated CNPJs for cnpj_list."
        ),
    )
    parser.add_argument(
        "--active-funds-only",
        action="store_true",
        help="With --fund-mode, keep only funds in 'EM FUNCIONAMENTO NORMAL' status.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help=(
            "Exit non-zero if any analyzer failed. Use in automation, where an "
            "empty report caused by a crash must not read as a clean result."
        ),
    )

    parser.add_argument("--enable-enrichment", action="store_true", help="Enable external context enrichment (Exa first).")
    parser.add_argument("--enrichment-provider", choices=["exa", "perplexity"], default="exa", help="Enrichment provider.")
    parser.add_argument("--enrichment-since-days", type=int, default=365, help="How far back to search.")
    parser.add_argument("--enrichment-max-results", type=int, default=8, help="Maximum results per query.")

    return parser


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def _selected_analyses(values: list[str] | None) -> set[str]:
    selected = set(values or ["all"])
    if "all" in selected:
        return set(ANALYSIS_CHOICES)
    return selected


def main(argv: list[str] | None = None) -> int:
    """Parse CLI arguments and execute the investigation pipeline."""
    args = _parse_args(argv)
    return run_investigation(args)


def _load_data(*, args: argparse.Namespace, config: Config, processor: DataProcessor) -> LoadedData:
    cadastro = None
    informe = None
    cda = None

    if args.cadastro:
        cadastro_path = Path(args.cadastro)
        cadastro = (
            processor.read_registro_fundo_classe(cadastro_path)
            if cadastro_path.name.startswith("registro_") or cadastro_path.is_dir()
            else processor.read_cadastro(cadastro_path)
        )
    else:
        # Prefer the RCVM 175 registry: the legacy cad_fi.csv joins to only 6.6%
        # of the funds the informe reports on, because reporting moved to the
        # class level and the old fund registrations were cancelled.
        registro = Path(args.public_data_dir) / "registro_classe.csv"
        if registro.exists():
            cadastro = processor.read_registro_fundo_classe(registro)
        else:
            maybe = _find_first(Path(args.public_data_dir), patterns=("cad_fi*.csv",))
            if maybe:
                logger.warning(
                    "Using legacy cad_fi.csv; it joins to a small fraction of "
                    "current funds. Download registro_fundo_classe.zip for full coverage."
                )
                cadastro = processor.read_cadastro(maybe)

    if args.informe:
        informe = _read_informe_any(Path(args.informe), processor=processor)
    else:
        processed_path = Path(args.processed_data_dir) / "reag_informe_diario_processed.csv"
        if processed_path.exists():
            informe = pd.read_csv(processed_path, sep=";", parse_dates=["DT_COMPTC"])
        else:
            maybe = _find_first(Path(args.public_data_dir), patterns=("inf_diario_fi_*.csv",))
            if maybe:
                informe = processor.read_informe_diario(maybe)

    if args.cda:
        cda = _read_cda_any(Path(args.cda), processor=processor)
    else:
        maybe = _find_first(Path(args.public_data_dir), patterns=("cda_fi_*.csv",))
        if maybe:
            cda = processor.read_cda(maybe)

    selected_cnpjs = resolve_fund_scope(args, cadastro)
    if selected_cnpjs:
        available_cnpjs: set[str] = set()
        for frame in (cadastro, informe, cda):
            if frame is None or frame.empty or "CNPJ_FUNDO" not in frame.columns:
                continue
            available_cnpjs.update(
                _normalize_selected_cnpjs(frame["CNPJ_FUNDO"].dropna().astype(str).tolist())
            )
        missing = [cnpj for cnpj in selected_cnpjs if cnpj not in available_cnpjs]
        if missing:
            logger.warning(
                "%d selected fund(s) not present in loaded data; continuing without them",
                len(missing),
            )

        if cadastro is not None and not cadastro.empty:
            cadastro = processor.filter_by_cnpj(cadastro, selected_cnpjs)
        if informe is not None and not informe.empty:
            before = len(informe)
            informe = processor.filter_by_cnpj(informe, selected_cnpjs)
            if informe.empty and before:
                # Silence here is the dangerous case: every downstream detector
                # would report nothing, which reads exactly like "no fraud".
                logger.error(
                    "Fund filter matched 0 of %d informe rows. None of the %d "
                    "selected funds reported in this period -- check that the "
                    "registry snapshot and the informe month line up.",
                    before, len(selected_cnpjs),
                )
        if cda is not None and not cda.empty:
            cda = processor.filter_by_cnpj(cda, selected_cnpjs)

    return LoadedData(cadastro=cadastro, informe=informe, cda=cda, selected_cnpjs=selected_cnpjs)


def resolve_fund_scope(
    args: argparse.Namespace, cadastro: pd.DataFrame | None
) -> list[str]:
    """Resolve which funds to investigate, as a list of normalized CNPJs.

    Combines two inputs:

    - ``--fund-mode`` / ``--fund-identifier``, resolved against the cadastro via
      :class:`~src.utils.fund_selector.FundSelector`. This is what makes the
      toolkit usable against any administrator or manager rather than only the
      case it was built for.
    - ``selected_cnpjs``, an explicit list supplied by the TUI.

    Both may be given; the result is their union. An empty list means no
    filtering, i.e. investigate every fund in the loaded data.

    Args:
        args: Parsed CLI arguments
        cadastro: Fund registry, needed to resolve names to CNPJs

    Returns:
        Normalized, de-duplicated CNPJs, or [] for "no filter"
    """
    explicit = _normalize_selected_cnpjs(getattr(args, "selected_cnpjs", []) or [])

    mode = getattr(args, "fund_mode", None)
    if not mode or mode == "all":
        return explicit

    identifier = getattr(args, "fund_identifier", None)

    if mode == "cnpj_list":
        if not identifier:
            logger.warning("--fund-mode cnpj_list requires --fund-identifier; ignoring")
            return explicit
        from_mode = _normalize_selected_cnpjs(
            [part.strip() for part in identifier.split(",")]
        )
        if not from_mode:
            logger.warning("No usable CNPJs in --fund-identifier %r", identifier)
        return list(dict.fromkeys(explicit + from_mode))

    if cadastro is None or cadastro.empty:
        logger.warning(
            "--fund-mode %s needs a cadastro to resolve names to CNPJs, "
            "but none was loaded; no fund filter applied",
            mode,
        )
        return explicit

    if not identifier:
        logger.warning("--fund-mode %s requires --fund-identifier; ignoring", mode)
        return explicit

    selector = FundSelector(cadastro)
    if mode == "administrator":
        matched = selector.select_by_administrator(identifier)
    elif mode == "manager":
        matched = selector.select_by_manager(identifier)
    else:  # pragma: no cover - argparse restricts the choices
        logger.warning("Unknown fund mode %r; ignoring", mode)
        return explicit

    if getattr(args, "active_funds_only", False):
        matched = selector.select_active_only(matched)

    if matched.empty or "CNPJ_FUNDO" not in matched.columns:
        logger.warning("No funds matched --fund-mode %s %r", mode, identifier)
        return explicit

    from_mode = _normalize_selected_cnpjs(
        matched["CNPJ_FUNDO"].dropna().astype(str).tolist()
    )
    logger.info("Fund scope: %d fund(s) matched %s %r", len(from_mode), mode, identifier)

    return list(dict.fromkeys(explicit + from_mode))


def _read_informe_any(path: Path, *, processor: DataProcessor) -> pd.DataFrame:
    if path.is_dir():
        files = sorted(path.glob("inf_diario_fi_*.csv"))
        if not files:
            return pd.DataFrame()
        dfs = [processor.read_informe_diario(p) for p in files[:24]]  # safety cap
        return pd.concat([df for df in dfs if not df.empty], ignore_index=True) if dfs else pd.DataFrame()
    # Heuristic: processed files are saved with ';'
    if path.name.endswith("_processed.csv"):
        return pd.read_csv(path, sep=";", parse_dates=["DT_COMPTC"])
    return processor.read_informe_diario(path)


def _read_cda_any(path: Path, *, processor: DataProcessor) -> pd.DataFrame:
    """Load CDA positions from a single CSV or a directory of monthly blocks.

    CVM ships the monthly CDA as one ZIP of many CSVs. The BLC_1..BLC_8 blocks
    hold positions split by asset class and are unioned here. Two kinds are
    excluded because they carry no asset-level position:

    - PL is a net-assets summary, one row per fund with no holdings at all.
    - CONFID holds positions the fund is permitted to withhold temporarily. It
      reports value but no quantity and no asset code, so its rows cannot
      support asset-level analysis and would collapse into a single phantom
      asset when grouped by a null CD_ATIVO.
    """
    if not path.is_dir():
        return processor.read_cda(path)

    files = [
        candidate
        for candidate in sorted(path.glob("cda_fi*.csv"))
        if "_PL_" not in candidate.name and "_CONFID_" not in candidate.name
    ]
    if not files:
        return pd.DataFrame()

    frames = [processor.read_cda(candidate) for candidate in files]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    logger.info(
        "Loaded %d CDA position rows from %d block(s)", len(combined), len(frames)
    )
    return combined


def _find_first(base_dir: Path, *, patterns: tuple[str, ...]) -> Path | None:
    if not base_dir.exists():
        return None
    candidates = []
    for pattern in patterns:
        candidates.extend(base_dir.glob(pattern))
        candidates.extend((base_dir / "cadastro").glob(pattern))
        candidates.extend((base_dir / "informe").glob(pattern))
        candidates.extend((base_dir / "cda").glob(pattern))
    return candidates[0] if candidates else None


def _normalize_cda_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Give CDA frames the column names analyzers expect.

    Delegates to DataProcessor so a CDA loaded through some other path gets the
    same treatment as one read by read_cda, rather than the two drifting apart.
    """
    return DataProcessor._normalize_cda_positions(df)


def _phantom_assets_by_fund(*, cda_df: pd.DataFrame, phantom_assets: pd.DataFrame) -> pd.DataFrame:
    if phantom_assets is None or phantom_assets.empty:
        return pd.DataFrame()
    if cda_df is None or cda_df.empty:
        return pd.DataFrame()
    if "CNPJ_FUNDO" not in cda_df.columns or "CD_ATIVO" not in cda_df.columns:
        return pd.DataFrame()
    flagged_assets = set(phantom_assets["asset_code"].astype(str).tolist())
    df = cda_df[cda_df["CD_ATIVO"].astype(str).isin(flagged_assets)].copy()
    if df.empty:
        return pd.DataFrame()

    # Aggregate per fund & asset.
    value_col = "VL_MERCADO" if "VL_MERCADO" in df.columns else None
    agg: dict[str, Any] = {}
    if value_col:
        agg["total_value_fund"] = (value_col, "sum")
    grouped = df.groupby(["CNPJ_FUNDO", "CD_ATIVO"]).agg(**agg).reset_index()
    grouped = grouped.rename(columns={"CD_ATIVO": "asset_code"})

    # Join risk/status/type from phantom_assets.
    risk_cols = ["asset_code", "asset_type", "status", "fraud_risk", "confidence", "issuer", "red_flags", "reason"]
    meta = phantom_assets[risk_cols].copy()
    return grouped.merge(meta, on="asset_code", how="left")


def _save_finding(df: pd.DataFrame, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if df is None:
        df = pd.DataFrame()
    df.to_csv(path, index=False)
    # Store relative reference for report readability (relative to run root).
    run_root = path.parent.parent
    try:
        return str(path.relative_to(run_root))
    except Exception:
        return str(path)


def _data_period(informe: pd.DataFrame | None) -> dict[str, Any]:
    """The date range the findings actually cover.

    A report that does not record its own period cannot be checked later. This
    repo learned that the hard way: reconstructing the window behind an earlier
    analysis meant sweeping two years of filings and matching aggregate totals
    to the cent, because nobody wrote the dates down.
    """
    if informe is None or informe.empty or "DT_COMPTC" not in informe.columns:
        return {}
    dates = pd.to_datetime(informe["DT_COMPTC"], errors="coerce").dropna()
    if dates.empty:
        return {}
    return {
        "start": dates.min().date().isoformat(),
        "end": dates.max().date().isoformat(),
        "reporting_days": int(dates.dt.date.nunique()),
        "funds_reporting": int(informe["CNPJ_FUNDO"].nunique())
        if "CNPJ_FUNDO" in informe.columns else None,
    }


def _write_run_metadata(
    *,
    output_dir: Path,
    run_id: str,
    args: argparse.Namespace,
    results: dict[str, pd.DataFrame],
    sources: dict[str, str],
    failures: list[str] | None = None,
    skipped: list[str] | None = None,
    selected_cnpjs: list[str] | None = None,
    data_period: dict[str, Any] | None = None,
) -> None:
    # Prefer the scope the load actually resolved -- otherwise a --fund-mode run
    # reports 0 funds while having filtered to that administrator's funds. Fall
    # back to the explicit list when the load reported none.
    if not selected_cnpjs:
        selected_cnpjs = _normalize_selected_cnpjs(getattr(args, "selected_cnpjs", []) or [])
    failures = failures or []
    skipped = skipped or []
    meta = {
        "run_id": run_id,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        # What the findings cover, not just when they were produced.
        "data_period": data_period or {},
        "args": vars(args),
        "scope": {
            "fund_filter_enabled": bool(selected_cnpjs),
            "selected_funds_count": len(selected_cnpjs),
            "selected_cnpjs": selected_cnpjs,
            "fund_mode": getattr(args, "fund_mode", None),
            "fund_identifier": getattr(args, "fund_identifier", None),
        },
        "outputs": {
            "sources": sources,
            "counts": {k: int(len(df)) for k, df in results.items()},
        },
        # An analyzer that crashed produces no findings, which is otherwise
        # indistinguishable from an analyzer that found nothing. Record both so
        # a run's own metadata says whether its coverage was complete.
        "execution": {
            "complete": not failures,
            "failed_analyzers": failures,
            "skipped_analyzers": skipped,
        },
        "disclaimer": [
            "Automated red flags are not proof of wrongdoing.",
            "Findings are prioritized leads requiring corroboration.",
        ]
        + (
            [
                f"INCOMPLETE RUN: {len(failures)} analyzer(s) failed "
                f"({', '.join(failures)}); absence of findings from them means "
                f"nothing."
            ]
            if failures
            else []
        ),
    }
    (output_dir / "summary.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def _run_context_enrichment(
    *,
    run_id: str,
    results: dict[str, pd.DataFrame],
    cadastro_df: pd.DataFrame | None,
    explain_max_entities: int,
    max_results: int,
    since_days: int,
    provider_name: str,
) -> dict[str, dict[str, Any]]:
    if provider_name != "exa":
        # Perplexity support is planned for a later phase.
        return {}

    provider = ExaProvider.from_env()
    if provider is None:
        return {}

    entities = _collect_flagged_entities(results=results, cadastro_df=cadastro_df)
    entities = entities[:explain_max_entities]

    contexts: dict[str, dict[str, Any]] = {}
    cache_root = Path("data/cache/enrichment") / run_id
    cache_root.mkdir(parents=True, exist_ok=True)

    for entity in entities:
        entity_id = entity["entity_id"]
        queries = _build_queries(entity)
        collected: list[SearchResult] = []
        for q in queries:
            try:
                collected.extend(provider.search(q, max_results=max_results, since_days=since_days))
            except Exception:
                continue

        # Deduplicate by URL deterministically.
        seen: set[str] = set()
        deduped: list[SearchResult] = []
        for r in collected:
            if r.url in seen:
                continue
            seen.add(r.url)
            deduped.append(r)

        raw_path = cache_root / f"{entity_id}.json"
        raw_payload = {
            "entity": entity,
            "queries": queries,
            "results": [r.__dict__ for r in deduped],
        }
        raw_path.write_text(json.dumps(raw_payload, indent=2), encoding="utf-8")

        contexts[entity_id] = build_context_section(deduped)

    return contexts


def _collect_flagged_entities(*, results: dict[str, pd.DataFrame], cadastro_df: pd.DataFrame | None) -> list[dict[str, Any]]:
    def normalize_digits(value: Any) -> str | None:
        if value is None:
            return None
        digits = "".join(ch for ch in str(value) if ch.isdigit())
        return digits if digits else None

    catalog: dict[str, dict[str, Any]] = {}
    fund_name_map = {}
    if cadastro_df is not None and not cadastro_df.empty and "CNPJ_FUNDO" in cadastro_df.columns:
        name_col = None
        for candidate in ("DENOM_SOCIAL", "DENOM_SOCIAL_FUNDO", "NOME_FUNDO", "NM_FUNDO"):
            if candidate in cadastro_df.columns:
                name_col = candidate
                break
        if name_col:
            for row in cadastro_df[["CNPJ_FUNDO", name_col]].dropna().itertuples(index=False):
                fund_name_map[normalize_digits(row[0])] = str(row[1]).strip()

    def add(entity_type: str, cnpj_like: Any, name: str | None = None) -> None:
        digits = normalize_digits(cnpj_like)
        entity_id = f"{entity_type}_{digits}" if digits else f"{entity_type}_{_stable_hash(str(cnpj_like) + (name or ''))}"
        if entity_id in catalog:
            return
        catalog[entity_id] = {
            "entity_id": entity_id,
            "entity_type": entity_type,
            "cnpj_digits": digits,
            "name": name,
        }

    # Fund-based outputs
    for key in ("flow_anomalies", "pl_drops", "runs", "divergences", "phantom_assets_by_fund"):
        df = results.get(key)
        if df is None or df.empty:
            continue
        if "CNPJ_FUNDO" in df.columns:
            for fund in df["CNPJ_FUNDO"].dropna().astype(str).unique():
                digits = normalize_digits(fund)
                add("FUND", digits, name=fund_name_map.get(digits))

    # Scheme outputs
    df = results.get("circular_flow")
    if df is not None and not df.empty:
        for row in df.to_dict(orient="records"):
            add("ADMINISTRATOR", row.get("admin_cnpj"))
            for fund in (row.get("CNPJ_FUNDO"), row.get("counterparty_cnpj")):
                if fund:
                    add("FUND", fund)

    df = results.get("layered_funds")
    if df is not None and not df.empty:
        for row in df.to_dict(orient="records"):
            add("ADMINISTRATOR", row.get("admin_cnpj"))
            add("FUND", row.get("holder_fund"))
            add("FUND", row.get("held_fund"))

    df = results.get("shell_networks")
    if df is not None and not df.empty:
        for row in df.to_dict(orient="records"):
            add("ADMINISTRATOR", row.get("admin_cnpj"))

    df = results.get("asset_inflation")
    if df is not None and not df.empty:
        for row in df.to_dict(orient="records"):
            add("FUND", row.get("fund_cnpj"))

    # Deterministic ordering: severity not known here, so sort by entity_id.
    return sorted(catalog.values(), key=lambda x: x["entity_id"])


def _build_queries(entity: dict[str, Any]) -> list[str]:
    entity_type = entity.get("entity_type")
    name = entity.get("name") or ""
    cnpj = entity.get("cnpj_digits") or ""
    if entity_type == "BANK":
        return [f"Banco {name} {cnpj} enforcement OR investigation OR CVM OR Banco Central"]
    if entity_type == "DISTRIBUTOR":
        return [f"{name} {cnpj} distribuição fundos CVM"]
    if entity_type == "FUND":
        return [f"{name} {cnpj} CVM fundo investigation OR administrator OR gestor"]
    if entity_type == "ADMINISTRATOR":
        return [f"{name} {cnpj} administradora fundos CVM investigation"]
    return [f"{name} {cnpj} investigation regulatory"]


def _stable_hash(text: str) -> str:
    import hashlib

    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


if __name__ == "__main__":
    raise SystemExit(main())
