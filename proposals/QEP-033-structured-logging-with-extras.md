# QEP-033: Structured Logging with Extras and Dual Sinks

**Status:** DRAFT
**Date:** 2026-02-20
**Author:** Claude Code
**Category:** Infrastructure

## Summary

Enhance logging to support structured output with contextual extras (settings, results, timings) written to a JSON file, while maintaining human-readable console output. This enables better observability, debugging, and audit trails for ODMR processing workflows.

## Motivation

Current logging in QDMpy is text-based and console-focused:
- No contextual information beyond message text
- No structured data (settings, fit results, performance metrics) attached to log entries
- Single log sink makes it difficult to parse logs programmatically
- Limited debugging capability for complex workflows with many pixels/frequencies

Structured logging with extras enables:
- **Searchable logs**: Filter by settings, fit metrics, image dimensions, etc.
- **Audit trails**: Record exact config used for each fit or processing step
- **Performance tracking**: Include timings, chi-squared values, memory usage
- **Debugging**: Context-rich error messages with full state attached
- **Integration**: Machine-readable JSON logs for external tooling (Jupyter, dashboards, etc.)

## GUI Integration Requirements

1. List the exact core API/data contract touchpoints used by `qdmpy-gui` (view-model calls, settings keys, map/result fields).
2. Define GUI state/settings migration behavior for any changed defaults, renamed keys, or persisted session/config data.
3. Specify expected user-facing behavior in the GUI for progress, warnings, and errors introduced by this QEP.
4. Include explicit GUI acceptance checks for this QEP scope: `load -> run action -> inspect outputs -> save/reload`, and verify no GUI-only workaround is required.
5. If impact is expected to be none, state the rationale and include a smoke check confirming no `qdmpy-gui` regression.

## Design

### 1. Dual-Sink Architecture
- **Console sink** (stdout): Human-readable format, INFO level, no serialization
- **File sink** (~/logs): Structured JSON format, DEBUG level, all context preserved

### 2. LoggingSettings Extension
```python
class LoggingSettings(BaseModel):
    """Settings for logging."""

    log_level: Literal['TRACE', 'DEBUG', 'INFO', 'SUCCESS', 'WARNING', 'ERROR', 'CRITICAL'] = 'INFO'
    log_file: str | None = None  # Legacy: optional persistent log

    # New: structured logging
    enable_structured_logging: bool = True
    structured_log_dir: str | None = None  # Defaults to ~/logs
```

### 3. Logger Configuration in `_configure_logging()`
```python
def _configure_logging(settings: QDMpySettings) -> None:
    """Configure loguru with console and structured JSON sink."""
    # Suppress noisy third-party loggers
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('h5py').setLevel(logging.WARNING)

    # Remove default handler
    logger.remove()

    # Console sink: human-readable, INFO level
    logger.add(
        sys.stdout,
        level=settings.logging.log_level,
        format='<level>{level: <8}</level> | {message}',
    )

    # Structured JSON sink: DEBUG level, all context
    if settings.logging.enable_structured_logging:
        log_dir = Path(settings.logging.structured_log_dir or Path.home() / 'logs')
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / 'qdmpy-{time:YYYY-MM-DD}.log'

        logger.add(
            str(log_file),
            level='DEBUG',
            format='{message}',
            serialize=True,  # JSON format
            rotation='10 MB',
            retention='7 days',
        )

    # Optional legacy file sink
    if settings.logging.log_file:
        logger.add(
            settings.logging.log_file,
            level=settings.logging.log_level,
            rotation='10 MB',
            retention='7 days',
        )
```

### 4. Logging with Extras at Key Analysis Stages

Loguru treats all kwargs as extras automatically—no `extra={}` wrapper needed.

#### ODMR Data Loading
```python
# odmr/manager.py
logger.info(
    'Loading ODMR data',
    loader_class=loader.__class__.__name__,
    file_count=len(files),
    raw_shape=tuple(raw_data.shape),
    n_polarity=raw_data.shape[0],
    n_frange=raw_data.shape[1],
    height=raw_data.shape[2],
    width=raw_data.shape[3],
    n_frequencies=raw_data.shape[4],
)
```

#### Data Processing (ODMRData → ODMRData)
```python
# measurement.py, in processing loop
import time
start = time.perf_counter()
input_shape = data.raw_data.shape
data = processor.process(data)
output_shape = data.raw_data.shape
elapsed_ms = (time.perf_counter() - start) * 1000

logger.info(
    'Processing step completed',
    processor_class=processor.__class__.__name__,
    processor_config=processor.model_dump(),  # Pydantic config
    input_shape=tuple(input_shape),
    output_shape=tuple(output_shape),
    processing_time_ms=int(elapsed_ms),
    data_modified=input_shape != output_shape,
)
```

#### Model Fitting (FitManager → FitResult)
```python
# fitting/manager.py - fit start
logger.info(
    'Fit started',
    model_name=self.model_name,
    n_pixels=n_pixels,
    n_frequencies=len(frequencies),
    constraints=self.constraints.model_dump(),
)

# After fit completes
import time
import numpy as np

chi2 = fit_result.chi2
success_mask = chi2 < 1e5
elapsed_ms = (fit_end_time - fit_start_time) * 1000

logger.info(
    'Fit completed',
    model_name=self.model_name,
    n_pixels=n_pixels,
    duration_ms=int(elapsed_ms),
    chi2_mean=float(chi2.mean()),
    chi2_median=float(np.median(chi2)),
    chi2_std=float(chi2.std()),
    chi2_min=float(chi2.min()),
    chi2_max=float(chi2.max()),
    successful_pixels=int(success_mask.sum()),
    success_rate=float(success_mask.sum() / n_pixels),
    result_shape=fit_result.center.shape,
)
```

#### B111 Field Computation
```python
# fitting/result.py, in b111 property
import numpy as np

b111_remanent = ...  # computed
b111_induced = ...   # computed

logger.info(
    'B111 field computed',
    n_pixels=self.center.shape[1] * self.center.shape[2],
    remanent_mean_µT=float(b111_remanent.mean()),
    remanent_std_µT=float(b111_remanent.std()),
    remanent_min_µT=float(b111_remanent.min()),
    remanent_max_µT=float(b111_remanent.max()),
    induced_mean_µT=float(b111_induced.mean()),
    induced_std_µT=float(b111_induced.std()),
    induced_min_µT=float(b111_induced.min()),
    induced_max_µT=float(b111_induced.max()),
)
```

#### Delta Resonance (Frequency Branch Splitting)
```python
# fitting/result.py, in delta_resonance property
delta_res = ...  # computed as xr.DataArray

neg_data = delta_res.sel(polarity='neg')
pos_data = delta_res.sel(polarity='pos')

logger.debug(
    'Delta resonance computed',
    shape=tuple(delta_res.shape),
    neg_mean_GHz=float(neg_data.mean()),
    neg_std_GHz=float(neg_data.std()),
    pos_mean_GHz=float(pos_data.mean()),
    pos_std_GHz=float(pos_data.std()),
)
```

#### Full Pipeline (Measurement.fit_odmr)
```python
# measurement.py, in fit_odmr() - start
import time

pipeline_start = time.perf_counter()

logger.info(
    'ODMR fit pipeline started',
    raw_shape=tuple(self.data.raw_data.shape),
    model_name=model_name,
    n_processors=len(self.data.processors),
    pixel_spacing_um=pixel_spacing,
)

# ... processing ...

proc_end = time.perf_counter()

# ... fitting ...

fit_end = time.perf_counter()

logger.info(
    'ODMR fit pipeline completed',
    total_duration_ms=int((fit_end - pipeline_start) * 1000),
    processing_duration_ms=int((proc_end - pipeline_start) * 1000),
    fitting_duration_ms=int((fit_end - proc_end) * 1000),
    final_shape=tuple(fit_result.center.shape),
    success_rate=float((fit_result.chi2 < 1e5).sum() / fit_result.chi2.size),
)
```

### 5. JSON Log Format (via `serialize=True`)

Each line is a valid JSON object with all kwargs in the `extra` field:
```json
{
  "timestamp": "2026-02-20T14:30:45.123456",
  "level": "INFO",
  "name": "QDMpy.odmr.manager",
  "message": "Loading ODMR data",
  "extra": {
    "loader_class": "MatlabLoader",
    "file_count": 5,
    "raw_shape": [2, 2, 1200, 1920, 50],
    "n_polarity": 2,
    "n_frange": 2,
    "height": 1200,
    "width": 1920,
    "n_frequencies": 50
  }
}
```

Query examples:
```bash
# Find all data loading events
jq '.[] | select(.message == "Loading ODMR data")' ~/logs/qdmpy-2026-02-20.log

# Get fit success rates
jq '.[] | select(.message == "Fit completed") | {model: .extra.model_name, success_rate: .extra.success_rate}' ~/logs/qdmpy-2026-02-20.log

# Find slow fits (>5s)
jq '.[] | select(.message == "Fit completed" and .extra.duration_ms > 5000)' ~/logs/qdmpy-2026-02-20.log

# Compare chi² distributions across models
jq '.[] | select(.message == "Fit completed") | {model: .extra.model_name, chi2_mean: .extra.chi2_mean, success_rate: .extra.success_rate}' ~/logs/qdmpy-2026-02-20.log
```

## Implementation Phases

### Phase 1: Infrastructure (this task)
- Update `LoggingSettings` in `settings.py`
- Update `_configure_logging()` to add structured JSON sink at ~/logs
- Create `~/logs` directory on first run
- No changes to existing logger calls (backward compatible)

### Phase 2: Instrumentation (follow-up PRs per analysis stage)
Systematically add kwargs to key logger calls organized by analysis stage:

1. **Data Loading** (`odmr/io.py`, `odmr/manager.py`)
   - Loader class, file count, raw data shape, dimensions

2. **Data Processing** (`measurement.py`, individual processors)
   - Processor class & config, input/output shapes, timing

3. **Fitting** (`fitting/manager.py`)
   - Model name, constraints, settings, chi² statistics, success rates

4. **Computed Results** (`fitting/result.py`)
   - B111 statistics, delta resonance, quality metrics

5. **Full Workflow** (`measurement.py` orchestration)
   - End-to-end timing, final stats, output location

### Phase 3: Testing (follow-up)
- Add fixtures to `tests/conftest.py` for capturing/parsing JSON logs
- Verify structured logs contain expected extras at each stage
- Add integration test to trace full pipeline through JSON logs

## Migration & Backward Compatibility

✅ **Fully backward compatible**:
- Existing logger calls work unchanged
- New `serialize=True` sink is additive (file only)
- Optional feature controlled by `enable_structured_logging` setting
- Legacy `log_file` setting still supported

Users can disable structured logging:
```toml
# ~/.config/QDMpy/settings.toml
[logging]
enable_structured_logging = false
```

## Risks & Mitigation

| Risk | Impact | Mitigation |
|------|--------|-----------|
| Disk space (JSON logs) | Medium | Default 10 MB rotation + 7-day retention |
| Serialization overhead | Low | JSON serialization only on file sink (async) |
| Sensitive data in extras | Medium | Document what can be logged; sanitize credentials in settings before logging |

## Success Criteria

- [ ] Phase 1 merged: settings.py updated, ~/logs created, dual sinks working
- [ ] Phase 2 complete: kwargs added at all 5 analysis stages
- [ ] Phase 3 complete: tests verify JSON structure and content
- [ ] No performance regression on fit operations (<5% overhead)
- [ ] Sample query scripts provided for common debugging tasks

## References

- [loguru documentation: extras](https://loguru.readthedocs.io/en/stable/api/logger.html#extras)
- [loguru documentation: serialize](https://loguru.readthedocs.io/en/stable/api/logger.html#serialize)
- [JSON log parsing best practices](https://www.kartar.net/2015/12/structured-logging/)
- QEP-030: Serializable Processor Pipeline (related: Pydantic models in extras)
- QEP-032: Module structure & orchestration (Measurement.fit_odmr)
