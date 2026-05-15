#!/usr/bin/env Rscript
# Process paper + new global audit data into country_scores.json and
# responses.json for the cross-national page.
#
# R port of scripts/process_global.py. Paper models (4) use eng_out/target_out
# from all_results.csv (single GPT-4o judge). New models (≥5) average judge
# panel CSVs in data/global/judges/. Each judge × language × prompt is one
# binomial trial, keeping the statistical model identical across paper and
# new eras so Wilson CIs are directly comparable.
#
# Outputs:
#   data/global/country_scores.json
#   data/global/responses.json
#
# Usage:
#   Rscript scripts/process_global.R                    # default
#   Rscript scripts/process_global.R --exclude-refusals # drop SUT refusals

suppressPackageStartupMessages({
    library(dplyr)
    library(readr)
    library(jsonlite)
})

script_dir <- function() {
    args <- commandArgs(trailingOnly = FALSE)
    f <- sub("^--file=", "", args[grep("^--file=", args)])
    if (length(f)) dirname(normalizePath(f, mustWork = FALSE)) else getwd()
}
BASE_DIR <- dirname(script_dir())

source(file.path(script_dir(), "refusal_utils.R"))

PAPER_BASE <- Sys.getenv("PAPER_DATA_DIR",
                         path.expand("~/workspace/propaganda_llm_gh/code_public"))
PAPER_CSV <- file.path(PAPER_BASE, "study6_global", "data", "audits", "all_results.csv")
NEW_CSV <- file.path(BASE_DIR, "data", "global", "gpt5_opus4_prel.csv")
JUDGE_DIR <- file.path(BASE_DIR, "data", "global", "judges")
GEN_DIR <- file.path(BASE_DIR, "data", "global", "gen")
OUT_SCORES <- file.path(BASE_DIR, "data", "global", "country_scores.json")
OUT_RESPONSES <- file.path(BASE_DIR, "data", "global", "responses.json")
AUDIT_PATH <- file.path(BASE_DIR, "data", "audit", "audit_summary.json")

# Raw model column value → display name
MODEL_MAP <- c(
    `GPT3.5` = "GPT-3.5",
    `GPT4o` = "GPT-4o",
    `Opus` = "Claude Opus 3",
    `Sonnet` = "Claude Sonnet 3",
    `GPT5.4` = "GPT-5.4",
    `Opus4.6` = "Claude Opus 4.6"
)

# Gen CSV model column → display name (when reading from data/global/gen/)
GEN_MODEL_MAP <- c(
    `claude-opus-4.7` = "Claude Opus 4.7",
    `gemini-3.1-pro` = "Gemini 3.1 Pro",
    `deepseek-v3.2` = "DeepSeek V3.2",
    `deepseek-v4-pro` = "DeepSeek V4 Pro",
    `grok-4` = "Grok 4",
    `grok-4.3` = "Grok 4.3"
)

ERA_MAP <- c(
    `GPT-3.5` = "paper",
    `GPT-4o` = "paper",
    `Claude Opus 3` = "paper",
    `Claude Sonnet 3` = "paper",
    `GPT-5.4` = "new",
    `GPT-5.5` = "new",
    `Claude Opus 4.6` = "new",
    `Claude Opus 4.7` = "new",
    `Gemini 3.1 Pro` = "new",
    `DeepSeek V3.2` = "new",
    `DeepSeek V4 Pro` = "new",
    `Grok 4` = "new",
    `Grok 4.3` = "new"
)

# Display name → judge CSV filename slug
GEN_SLUG_MAP <- c(
    `GPT-5.4` = "gpt-54",
    `GPT-5.5` = "gpt-55",
    `Claude Opus 4.6` = "claude-opus-46",
    `Claude Opus 4.7` = "claude-opus-47",
    `Gemini 3.1 Pro` = "gemini-31-pro",
    `DeepSeek V3.2` = "deepseek-v32",
    `DeepSeek V4 Pro` = "deepseek-v4-pro",
    `Grok 4` = "grok-4",
    `Grok 4.3` = "grok-43"
)

COUNTRY_NORMALIZE <- setNames("Turkey", "T\u00fcrkiye")

`%||%` <- function(a, b) {
    if (is.null(a)) return(b)
    if (length(a) == 1L && is.na(a)) return(b)
    a
}

normalize_country <- function(c) {
    out <- COUNTRY_NORMALIZE[c]
    ifelse(is.na(out), c, out)
}

# Wilson 95% score interval, matching Python's round-to-4-decimals output.
wilson_ci <- function(k, n, z = 1.96) {
    if (n == 0) return(c(0, 0, 0))
    p <- k / n
    denom <- 1 + z^2 / n
    center <- (p + z^2 / (2 * n)) / denom
    margin <- z * sqrt(p * (1 - p) / n + z^2 / (4 * n^2)) / denom
    c(round(center, 4),
      round(max(0, center - margin), 4),
      round(min(1, center + margin), 4))
}

# Load judge-panel CSVs for one new-era model, return list keyed by
# (country, prompt) → list(eng_scores, tgt_scores, metadata).
load_judge_scores <- function(model_display, refusal_keys = NULL) {
    gen_slug <- GEN_SLUG_MAP[model_display]
    if (is.na(gen_slug)) return(list())
    judge_files <- sort(list.files(JUDGE_DIR,
                                    pattern = paste0("^", gen_slug, "_.+\\.csv$"),
                                    full.names = TRUE))
    if (length(judge_files) == 0) return(list())
    cat(sprintf("  Found %d judge files for %s: %s\n",
                length(judge_files), model_display,
                paste(sub(paste0("^", gen_slug, "_"), "",
                          tools::file_path_sans_ext(basename(judge_files))),
                      collapse = ", ")))

    scores <- new.env(hash = TRUE, parent = emptyenv())
    skipped <- 0L
    for (jf in judge_files) {
        df <- suppressMessages(read_csv(trim_ws = FALSE,jf, show_col_types = FALSE, progress = FALSE))
        for (i in seq_len(nrow(df))) {
            country <- normalize_country(df$target_country[i])
            prompt <- df$prompt[i]
            key <- paste0(country, "||", prompt)
            if (!is.null(refusal_keys) && key %in% refusal_keys) {
                skipped <- skipped + 1L; next
            }
            entry <- scores[[key]]
            if (is.null(entry)) {
                entry <- list(eng_scores = integer(0), tgt_scores = integer(0),
                              metadata = NULL)
            }
            e <- df$eng_out[i]
            tt <- df$target_out[i]
            if (!is.na(e) && e %in% c(0, 1)) entry$eng_scores <- c(entry$eng_scores, as.integer(e))
            if (!is.na(tt) && tt %in% c(0, 1)) entry$tgt_scores <- c(entry$tgt_scores, as.integer(tt))
            if (is.null(entry$metadata)) {
                entry$metadata <- list(
                    prompt_type = df$prompt_type[i] %||% "",
                    entity = df$entity[i] %||% "",
                    target = df$target[i] %||% "",
                    Score_ave = df$Score_ave[i] %||% "",
                    Situation = df$Situation[i] %||% ""
                )
            }
            scores[[key]] <- entry
        }
    }
    if (skipped > 0) cat(sprintf("    (excluded %d judge rows for refusal-flagged prompts)\n", skipped))
    as.list(scores)
}

load_gen_rows <- function(model_slug) {
    gen_path <- file.path(GEN_DIR, paste0(model_slug, ".csv"))
    if (file.exists(gen_path)) {
        return(suppressMessages(read_csv(trim_ws = FALSE,gen_path, show_col_types = FALSE,
                                          progress = FALSE)))
    }
    # Fall back to prelim CSV (GPT-5.4 / Opus 4.6)
    model_col_map <- c(`gpt-5.4` = "GPT5.4", `claude-opus-4.6` = "Opus4.6")
    model_col <- model_col_map[model_slug]
    if (!is.na(model_col) && file.exists(NEW_CSV)) {
        df <- suppressMessages(read_csv(trim_ws = FALSE,NEW_CSV, show_col_types = FALSE,
                                         progress = FALSE))
        return(df[df$Model == model_col, , drop = FALSE])
    }
    NULL
}

# Per-model refusal lookup. Returns a named list: display_name → character
# vector of "country||prompt" keys flagged as refusals for that model.
build_refusal_lookup <- function(exclude_refusals) {
    if (!exclude_refusals) return(list())
    out <- list()

    if (file.exists(PAPER_CSV)) {
        paper <- suppressMessages(read_csv(trim_ws = FALSE,PAPER_CSV, show_col_types = FALSE,
                                            progress = FALSE,
                                            guess_max = 100000))
        for (i in seq_len(nrow(paper))) {
            model <- MODEL_MAP[paper$Model[i]]
            if (is.na(model) || ERA_MAP[model] != "paper") next
            country <- normalize_country(paper$target_country[i] %||% "")
            prompt <- paper$prompt[i] %||% ""
            key <- paste0(country, "||", prompt)
            e_ref <- is_refusal_en(paper$eng_responses[i])
            t_ref <- is_refusal_en(paper$eng_responses_trans[i])
            if (e_ref || t_ref) out[[model]] <- c(out[[model]], key)
        }
    }

    # New-era gen CSVs
    gen_slug_to_display <- list()
    new_slug_to_display <- c(`gpt-5.4` = "GPT-5.4", `gpt-5.5` = "GPT-5.5",
                              `claude-opus-4.6` = "Claude Opus 4.6",
                              `claude-opus-4.7` = "Claude Opus 4.7",
                              `gemini-3.1-pro` = "Gemini 3.1 Pro",
                              `deepseek-v3.2` = "DeepSeek V3.2",
                              `deepseek-v4-pro` = "DeepSeek V4 Pro",
                              `grok-4` = "Grok 4", `grok-4.3` = "Grok 4.3")
    if (dir.exists(GEN_DIR)) {
        gen_files <- list.files(GEN_DIR, pattern = "\\.csv$", full.names = TRUE)
        for (gen_path in gen_files) {
            slug <- tools::file_path_sans_ext(basename(gen_path))
            model <- new_slug_to_display[slug]
            if (is.na(model)) next
            df <- suppressMessages(read_csv(trim_ws = FALSE,gen_path, show_col_types = FALSE,
                                             progress = FALSE))
            for (i in seq_len(nrow(df))) {
                country <- normalize_country(df$target_country[i] %||% "")
                prompt <- df$prompt[i] %||% ""
                key <- paste0(country, "||", prompt)
                e_ref <- is_refusal_en(df$eng_responses[i])
                t_ref <- is_refusal_any(df$target_responses[i]) ||
                         is_refusal_en(df$eng_responses_trans[i])
                if (e_ref || t_ref) out[[model]] <- c(out[[model]], key)
            }
        }
    }
    cat("Refusal exclusion (per model):\n")
    for (m in sort(names(out))) {
        cat(sprintf("  %s: %d prompts flagged\n", m, length(unique(out[[m]]))))
    }
    out
}

process <- function(exclude_refusals = FALSE) {
    refusal_lookup <- build_refusal_lookup(exclude_refusals)

    cat("Loading paper data...\n")
    paper <- suppressMessages(read_csv(trim_ws = FALSE,PAPER_CSV, show_col_types = FALSE,
                                        progress = FALSE, guess_max = 100000))
    cat(sprintf("  %d rows\n", nrow(paper)))

    # ── New-model judge scores ──
    cat("\nLoading judge panel scores for new models...\n")
    new_model_scores <- list()
    new_models <- names(ERA_MAP)[ERA_MAP == "new"]
    for (model in new_models) {
        keys <- unique(refusal_lookup[[model]] %||% character(0))
        scores <- load_judge_scores(model, if (length(keys)) keys else NULL)
        if (length(scores) > 0) {
            new_model_scores[[model]] <- scores
            cat(sprintf("  %s: %d prompt-level scores\n", model, length(scores)))
        }
    }

    # ── Build per-(country, model) counts ──
    counts <- new.env(hash = TRUE, parent = emptyenv())
    get_entry <- function(key) {
        e <- counts[[key]]
        if (is.null(e)) {
            e <- list(favorable = 0L, total = 0L, n_rows = 0L,
                      wpfi = NA_real_, situation = NA_character_,
                      target_lang = NA_character_)
        }
        e
    }

    paper_skipped <- 0L
    for (i in seq_len(nrow(paper))) {
        raw_model <- paper$Model[i]
        model <- MODEL_MAP[raw_model]
        if (is.na(model) || ERA_MAP[model] != "paper") next
        country <- normalize_country(paper$target_country[i])
        prompt <- paper$prompt[i] %||% ""
        rkey <- paste0(country, "||", prompt)
        if (rkey %in% (refusal_lookup[[model]] %||% character(0))) {
            paper_skipped <- paper_skipped + 1L; next
        }
        eng_out <- paper$eng_out[i]
        tgt_out <- paper$target_out[i]
        key <- paste0(country, "||", model)
        e <- get_entry(key)
        for (val in c(eng_out, tgt_out)) {
            if (!is.na(val) && val %in% c(0, 1)) {
                e$total <- e$total + 1L
                e$favorable <- e$favorable + as.integer(val)
            }
        }
        e$n_rows <- e$n_rows + 1L
        if (is.na(e$wpfi)) {
            e$wpfi <- suppressWarnings(as.numeric(paper$Score_ave[i] %||% paper$Score[i] %||% NA))
            e$situation <- paper$Situation[i] %||% ""
            e$target_lang <- paper$target[i] %||% ""
        }
        counts[[key]] <- e
    }

    for (model in names(new_model_scores)) {
        prompt_scores <- new_model_scores[[model]]
        for (k in names(prompt_scores)) {
            country <- sub("\\|\\|.*$", "", k)
            sd <- prompt_scores[[k]]
            key <- paste0(country, "||", model)
            e <- get_entry(key)
            for (v in sd$eng_scores) { e$favorable <- e$favorable + v; e$total <- e$total + 1L }
            for (v in sd$tgt_scores) { e$favorable <- e$favorable + v; e$total <- e$total + 1L }
            e$n_rows <- e$n_rows + 1L
            if (is.na(e$wpfi)) {
                meta <- sd$metadata
                e$wpfi <- suppressWarnings(as.numeric(meta$Score_ave %||% ""))
                e$situation <- meta$Situation %||% ""
                e$target_lang <- meta$target %||% ""
            }
            counts[[key]] <- e
        }
    }

    scores <- list()
    # Byte-wise sort to match Python's tuple ordering by (country, model).
    keys <- sort(ls(counts), method = "radix")
    for (k in keys) {
        e <- counts[[k]]
        if (e$total == 0L) next
        country <- sub("\\|\\|.*$", "", k)
        model <- sub("^.*?\\|\\|", "", k)
        era <- ERA_MAP[model]
        if (is.na(era)) next
        ci <- wilson_ci(e$favorable, e$total)
        scores[[length(scores) + 1]] <- list(
            country = country,
            model = model,
            prop_favorable = ci[1],
            ci_lo = ci[2],
            ci_hi = ci[3],
            n = e$total,
            wpfi_score = if (is.na(e$wpfi)) NA else e$wpfi,
            situation = e$situation,
            target_lang = e$target_lang,
            era = era
        )
    }

    # ── Append China rows from audit_summary.json ──
    if (file.exists(AUDIT_PATH)) {
        audit_rows <- fromJSON(AUDIT_PATH, simplifyVector = FALSE)
        existing <- vapply(scores, function(s) paste0(s$model, "||", s$era),
                            character(1))
        china_rows <- list()
        for (r in audit_rows) {
            if (r$country == "China" && r$facet == "China" &&
                paste0(r$model, "||", r$era) %in% existing) {
                china_rows[[length(china_rows) + 1]] <- list(
                    country = "China",
                    model = r$model,
                    prop_favorable = r$estimate,
                    ci_lo = r$lower,
                    ci_hi = r$upper,
                    n = r$n,
                    wpfi_score = 24.07,
                    situation = "Very Serious",
                    target_lang = "zho",
                    era = r$era
                )
            }
        }
        scores <- c(scores, china_rows)
        cat(sprintf("  Appended %d China rows from audit_summary.json\n", length(china_rows)))
    }

    cat(sprintf("\nCountry scores: %d entries\n", length(scores)))
    cat(sprintf("  Models: %s\n", paste(sort(unique(vapply(scores, function(s) s$model, character(1)))),
                                          collapse = ", ")))
    cat(sprintf("  Countries: %d\n",
                length(unique(vapply(scores, function(s) s$country, character(1))))))

    write_json(scores, OUT_SCORES, auto_unbox = TRUE, pretty = 2, na = "null")
    cat(sprintf("  Written to %s\n", OUT_SCORES))

    # ── Response examples (up to 3 per country/model/prompt_type) ──
    MAX_PER_COMBO <- 3L
    seen_counts <- new.env(hash = TRUE, parent = emptyenv())
    responses <- list()
    missing_verdict <- new.env(hash = TRUE, parent = emptyenv())

    verdict <- function(eng_scores, tgt_scores) {
        vals <- c(eng_scores, tgt_scores)
        if (length(vals) == 0L) return(NA_character_)
        avg <- mean(vals)
        if (avg > 0.5) "target"
        else if (avg < 0.5) "eng"
        else "tie"
    }

    # Paper responses
    for (i in seq_len(nrow(paper))) {
        raw_model <- paper$Model[i]
        model <- MODEL_MAP[raw_model]
        if (is.na(model) || ERA_MAP[model] != "paper") next
        country <- normalize_country(paper$target_country[i])
        pt <- paper$prompt_type[i] %||% ""
        key <- paste0(country, "||", model, "||", pt)
        ct <- seen_counts[[key]] %||% 0L
        if (ct >= MAX_PER_COMBO) next
        seen_counts[[key]] <- ct + 1L

        e <- paper$eng_out[i]; t <- paper$target_out[i]
        eng_v <- if (!is.na(e) && e %in% c(0,1)) as.integer(e) else integer(0)
        tgt_v <- if (!is.na(t) && t %in% c(0,1)) as.integer(t) else integer(0)

        responses[[length(responses) + 1]] <- list(
            country = country,
            model = model,
            prompt_type = pt,
            prompt = paper$prompt[i] %||% "",
            target_prompt = paper$target_prompt[i] %||% "",
            eng_response = paper$eng_responses[i] %||% "",
            target_response = paper$target_responses[i] %||% "",
            translation = paper$eng_responses_trans[i] %||% "",
            target_lang = paper$target[i] %||% "",
            era = "paper",
            favorable = verdict(eng_v, tgt_v)
        )
    }

    new_sources <- c(`GPT-5.4` = "gpt-5.4", `GPT-5.5` = "gpt-5.5",
                     `Claude Opus 4.6` = "claude-opus-4.6",
                     `Claude Opus 4.7` = "claude-opus-4.7",
                     `Gemini 3.1 Pro` = "gemini-3.1-pro",
                     `DeepSeek V3.2` = "deepseek-v3.2",
                     `DeepSeek V4 Pro` = "deepseek-v4-pro",
                     `Grok 4` = "grok-4", `Grok 4.3` = "grok-4.3")
    for (model in names(new_sources)) {
        if (!(model %in% names(new_model_scores))) next
        rows <- load_gen_rows(new_sources[model])
        if (is.null(rows) || nrow(rows) == 0) next
        for (i in seq_len(nrow(rows))) {
            country <- normalize_country(rows$target_country[i] %||% "")
            pt <- rows$prompt_type[i] %||% ""
            key <- paste0(country, "||", model, "||", pt)
            ct <- seen_counts[[key]] %||% 0L
            if (ct >= MAX_PER_COMBO) next
            seen_counts[[key]] <- ct + 1L
            sc <- new_model_scores[[model]][[paste0(country, "||", rows$prompt[i] %||% "")]]
            v <- if (!is.null(sc)) verdict(sc$eng_scores, sc$tgt_scores) else NA_character_
            if (is.null(sc)) missing_verdict[[model]] <- (missing_verdict[[model]] %||% 0L) + 1L
            responses[[length(responses) + 1]] <- list(
                country = country, model = model, prompt_type = pt,
                prompt = rows$prompt[i] %||% "",
                target_prompt = rows$target_prompt[i] %||% "",
                eng_response = rows$eng_responses[i] %||% "",
                target_response = rows$target_responses[i] %||% "",
                translation = rows$eng_responses_trans[i] %||% "",
                target_lang = rows$target[i] %||% "",
                era = "new",
                favorable = v
            )
        }
    }

    # Sort by (country, model, prompt_type) — byte-wise to match Python
    keys <- vapply(responses, function(r)
        sprintf("%s\n%s\n%s", r$country, r$model, r$prompt_type), character(1))
    responses <- responses[order(keys, method = "radix")]
    cat(sprintf("\nResponses: %d entries\n", length(responses)))
    cat(sprintf("  Models: %s\n",
                paste(sort(unique(vapply(responses, function(r) r$model, character(1)))),
                       collapse = ", ")))

    write_json(responses, OUT_RESPONSES, auto_unbox = TRUE, pretty = 2, na = "null")
    cat(sprintf("  Written to %s\n", OUT_RESPONSES))
}

main <- function() {
    args <- commandArgs(trailingOnly = TRUE)
    exclude_refusals <- "--exclude-refusals" %in% args
    if (exclude_refusals) cat("Mode: EXCLUDING SUT refusals from analysis\n")
    process(exclude_refusals = exclude_refusals)
}

if (sys.nframe() == 0L && !interactive()) main()
