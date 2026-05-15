#!/usr/bin/env Rscript
# Process Study 4 audit CSV files into summary JSON for the interactive chart.
#
# R port of scripts/process_study4_audit.py. Reads paper + new-era _res.csv
# files, computes per-(model, country) proportion of responses judged "more
# favorable" (Y = 1), with normal-approximation 95% CIs that match the paper.
#
# Output: data/audit/audit_summary.json
#
# Usage:
#   Rscript scripts/process_study4_audit.R                    # default
#   Rscript scripts/process_study4_audit.R --exclude-refusals # drop SUT refusals

suppressPackageStartupMessages({
    library(dplyr)
    library(readr)
    library(jsonlite)
    library(tidyr)
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
PAPER_DIR <- file.path(PAPER_BASE, "study4_production_model_audit", "data")
NEW_DIR <- file.path(BASE_DIR, "data", "study4")
OUT_PATH <- file.path(BASE_DIR, "data", "audit", "audit_summary.json")

PAPER_MODELS <- tibble::tribble(
    ~slug,    ~name,             ~era,
    "gpt3",   "GPT-3.5",         "paper",
    "gpt4",   "GPT-4o",          "paper",
    "opus",   "Claude Opus 3",   "paper",
    "sonnet", "Claude Sonnet 3", "paper"
)

NEW_MODELS <- tibble::tribble(
    ~slug,                       ~name,              ~era,
    "gpt-5-4",                   "GPT-5.4",          "new",
    "gpt-5-5",                   "GPT-5.5",          "new",
    "claude-opus-4-6",           "Claude Opus 4.6",  "new",
    "claude-opus-4-7",           "Claude Opus 4.7",  "new",
    "gemini-3-1-pro",            "Gemini 3.1 Pro",   "new",
    "deepseek-v3-2-speciale",    "DeepSeek V3.2",    "new",
    "deepseek-v4-pro",           "DeepSeek V4 Pro",  "new",
    "grok-4",                    "Grok 4",           "new",
    "grok-4-3",                  "Grok 4.3",         "new"
)

PROMPT_TYPES <- c("country", "leader", "inst")

# Country code → (display name, facet). Excludes DE to match paper.
COUNTRY_MAP <- list(
    US = list(name = "United States", facet = "Baseline"),
    UK = list(name = "United Kingdom", facet = "Baseline"),
    CN = list(name = "China",         facet = "China"),
    NK = list(name = "North Korea",   facet = "Spillover"),
    RU = list(name = "Russia",        facet = "Spillover")
)

# Per-prompt refusal mask keyed by the English prompt text. A prompt is flagged
# if the SUT refused in EITHER the Chinese-prompt or English-prompt response.
# (The result CSVs use the English prompt as row id; see
# run_judge_panel.build_comparison_df.)
build_refusal_mask <- function(gen_path) {
    if (!file.exists(gen_path)) return(NULL)
    df <- suppressMessages(read_csv(gen_path, show_col_types = FALSE))
    cn <- df %>% filter(language == "cn")
    en <- df %>% filter(language == "en")
    n <- min(nrow(cn), nrow(en))
    if (n == 0) return(NULL)
    en_prompts <- en$prompt[seq_len(n)]
    cn_refused <- is_refusal_any_v(cn$response_cn[seq_len(n)])
    en_refused <- is_refusal_any_v(en$response_en[seq_len(n)])
    setNames(cn_refused | en_refused, en_prompts)
}

# Load all prompt-type CSVs for one model, return combined data frame.
# For new-era models, average Y across panel judges (files matching
# {pt}_{slug}_res_{judge}.csv); for paper-era, use the single _res.csv.
load_model_data <- function(base_dir, slug, model_name, era,
                            exclude_refusals = FALSE, gen_base_dir = NULL) {
    frames <- list()
    refusal_dropped <- 0L
    for (pt in PROMPT_TYPES) {
        refusal_mask <- NULL
        if (exclude_refusals) {
            gen_dir <- gen_base_dir %||% base_dir
            refusal_mask <- build_refusal_mask(file.path(gen_dir, paste0(pt, "_", slug, ".csv")))
        }

        panel_paths <- sort(list.files(
            base_dir, pattern = paste0("^", pt, "_", slug, "_res_.+\\.csv$"),
            full.names = TRUE))

        if (era == "new" && length(panel_paths) > 0) {
            judge_frames <- list()
            for (path in panel_paths) {
                df <- suppressMessages(read_csv(path, show_col_types = FALSE))
                if (!all(c("Y_cn", "Y_en") %in% names(df))) next
                keep_cols <- c("country", "Y_cn", "Y_en")
                if ("prompt" %in% names(df)) keep_cols <- c("prompt", keep_cols)
                df <- df[, keep_cols, drop = FALSE]
                df$prompt_type <- pt
                if (!is.null(refusal_mask) && "prompt" %in% names(df)) {
                    before <- nrow(df)
                    mask <- refusal_mask[df$prompt]
                    mask[is.na(mask)] <- FALSE
                    df <- df[!mask, , drop = FALSE]
                    refusal_dropped <- refusal_dropped + (before - nrow(df))
                }
                df$prompt <- NULL
                judge_frames[[length(judge_frames) + 1]] <- df
            }
            if (length(judge_frames) > 0) {
                frames[[length(frames) + 1]] <- bind_rows(judge_frames)
            }
        } else {
            path <- file.path(base_dir, paste0(pt, "_", slug, "_res.csv"))
            if (!file.exists(path)) next
            df <- suppressMessages(read_csv(path, show_col_types = FALSE))
            if (!all(c("Y_cn", "Y_en") %in% names(df))) next
            keep_cols <- c("country", "Y_cn", "Y_en")
            if ("prompt" %in% names(df)) keep_cols <- c("prompt", keep_cols)
            df <- df[, keep_cols, drop = FALSE]
            df$prompt_type <- pt
            if (!is.null(refusal_mask) && "prompt" %in% names(df)) {
                before <- nrow(df)
                mask <- refusal_mask[df$prompt]
                mask[is.na(mask)] <- FALSE
                df <- df[!mask, , drop = FALSE]
                refusal_dropped <- refusal_dropped + (before - nrow(df))
            }
            df$prompt <- NULL
            frames[[length(frames) + 1]] <- df
        }
    }
    if (length(frames) == 0) return(NULL)
    combined <- bind_rows(frames)
    combined$model <- model_name
    combined$era <- era
    if (exclude_refusals && refusal_dropped > 0) {
        cat(sprintf("    (excluded %d refusal rows)\n", refusal_dropped))
    }
    combined
}

# Per-(model, country) proportion favorable with normal-approximation CI.
# Converts Y from {-1, 1} to binary {0, 1} via max(Y, 0), stacks CN + EN
# responses, requires n >= 20 per cell (matches Python).
compute_summary <- function(df) {
    records <- list()
    groups <- df %>%
        group_by(model, country, era) %>%
        group_split()
    for (grp in groups) {
        country_code <- grp$country[1]
        if (!(country_code %in% names(COUNTRY_MAP))) next
        cm <- COUNTRY_MAP[[country_code]]
        y_cn <- pmax(grp$Y_cn[!is.na(grp$Y_cn)], 0)
        y_en <- pmax(grp$Y_en[!is.na(grp$Y_en)], 0)
        y_all <- c(y_cn, y_en)
        n <- length(y_all)
        if (n < 20) next
        p <- mean(y_all)
        se <- sqrt(p * (1 - p) / n)
        records[[length(records) + 1]] <- list(
            model = grp$model[1],
            country = cm$name,
            country_code = country_code,
            facet = cm$facet,
            estimate = round(p, 4),
            se = round(se, 4),
            lower = round(max(p - 1.96 * se, 0), 4),
            upper = round(min(p + 1.96 * se, 1), 4),
            n = n,
            era = grp$era[1]
        )
    }
    records
}

`%||%` <- function(a, b) if (!is.null(a)) a else b

main <- function() {
    args <- commandArgs(trailingOnly = TRUE)
    exclude_refusals <- "--exclude-refusals" %in% args
    if (exclude_refusals) cat("Mode: EXCLUDING SUT refusals from analysis\n")

    all_frames <- list()

    for (i in seq_len(nrow(PAPER_MODELS))) {
        row <- PAPER_MODELS[i, ]
        df <- load_model_data(PAPER_DIR, row$slug, row$name, row$era,
                              exclude_refusals = exclude_refusals,
                              gen_base_dir = PAPER_DIR)
        if (!is.null(df)) {
            cat(sprintf("  %s: %d rows\n", row$name, nrow(df)))
            all_frames[[length(all_frames) + 1]] <- df
        }
    }
    for (i in seq_len(nrow(NEW_MODELS))) {
        row <- NEW_MODELS[i, ]
        df <- load_model_data(NEW_DIR, row$slug, row$name, row$era,
                              exclude_refusals = exclude_refusals,
                              gen_base_dir = NEW_DIR)
        if (!is.null(df)) {
            cat(sprintf("  %s: %d rows\n", row$name, nrow(df)))
            all_frames[[length(all_frames) + 1]] <- df
        } else {
            cat(sprintf("  %s: no data yet\n", row$name))
        }
    }
    if (length(all_frames) == 0) {
        cat("No data found!\n"); return(invisible())
    }
    combined <- bind_rows(all_frames)
    summary <- compute_summary(combined)
    # Sort: paper first, then by model name, then country. Use multi-key
    # radix order so byte-wise comparison runs on each field independently
    # (a concatenated key with a separator would let the separator interfere
    # with the lexicographic comparison of model names).
    era_order <- c(paper = 0L, new = 1L)
    era_keys <- vapply(summary, function(r) era_order[r$era] %||% 2L, integer(1))
    model_keys <- vapply(summary, function(r) r$model, character(1))
    country_keys <- vapply(summary, function(r) r$country, character(1))
    summary <- summary[order(era_keys, model_keys, country_keys, method = "radix")]

    dir.create(dirname(OUT_PATH), recursive = TRUE, showWarnings = FALSE)
    write_json(summary, OUT_PATH, auto_unbox = TRUE, pretty = 2, na = "null")
    cat(sprintf("\nWrote %d records to %s\n", length(summary), OUT_PATH))

    models <- sort(unique(vapply(summary, function(r) r$model, character(1))))
    for (m in models) {
        n <- sum(vapply(summary, function(r) r$model == m, logical(1)))
        cat(sprintf("  %s: %d country groups\n", m, n))
    }
}

if (sys.nframe() == 0L && !interactive()) main()
