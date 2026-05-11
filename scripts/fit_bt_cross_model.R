#!/usr/bin/env Rscript
# Bradley-Terry fits on cross-model audit comparisons.
#
# Reads all data/cross_model_audit/*.csv (pairwise judge outcomes), and for
# each (focal_country, language, judge) stratum fits a Bradley-Terry model
# estimating each SUT's favorability ability (log-odds). Writes
# data/cross_model_audit/bt_scores.json for the OJS chart.
#
# Output JSON schema: list of
#   {model, focal_country, language, judge, bt_logit, ci_lo, ci_hi, n_pairs}
# Scores are anchored with GPT-5.5 = 0 (reference).
#
# Usage: Rscript scripts/fit_bt_cross_model.R

suppressPackageStartupMessages({
    library(BradleyTerry2)
    library(dplyr)
    library(jsonlite)
    library(readr)
})

# Country code → display
COUNTRY_DISPLAY <- c(
    CN = "China", RU = "Russia", NK = "North Korea",
    US = "United States", UK = "United Kingdom", DE = "Germany"
)

# Reference SUT — anchored at 0 in BT log-odds
REFERENCE_MODEL <- "gpt-5.5"

read_all_comparisons <- function(dir = "data/cross_model_audit") {
    files <- list.files(dir, pattern = "_vs_.*\\.csv$", full.names = TRUE)
    stopifnot(length(files) > 0)
    rows <- lapply(files, function(f) {
        df <- suppressMessages(read_csv(f, show_col_types = FALSE))
        # Filename: {model_a}_vs_{model_b}_{judge}.csv with dots already stripped
        stem <- tools::file_path_sans_ext(basename(f))
        parts <- strsplit(stem, "_vs_")[[1]]
        right <- strsplit(parts[2], "_")[[1]]
        # The judge slug is the LAST underscore-separated segment(s).
        # Find by matching against known judge slugs.
        judge_slug <- tail(right, 1)
        # multi-piece judges like "gpt-oss-120b" are already single token (dashes)
        df$judge <- judge_slug
        df
    })
    bind_rows(rows)
}

fit_one_stratum <- function(df, focal_country, language, judge) {
    # judge == "all" pools observations across all judges (each judge × prompt
    # × pair contributes one binomial outcome — double the n for a single fit)
    if (identical(judge, "all")) {
        sub <- df %>%
            filter(country == focal_country, language == !!language,
                   !is.na(winner), winner %in% c("model_a", "model_b"))
    } else {
        sub <- df %>%
            filter(country == focal_country, language == !!language,
                   judge == !!judge,
                   !is.na(winner), winner %in% c("model_a", "model_b"))
    }
    if (nrow(sub) < 10) return(NULL)

    # BradleyTerry2 expects (player1, player2, outcome) where outcome is
    # 0/1 for player1 winning OR a factor; cleanest: counts table.
    sub <- sub %>% mutate(
        winner_model = ifelse(winner == "model_a", as.character(model_a),
                              as.character(model_b)),
        loser_model = ifelse(winner == "model_a", as.character(model_b),
                             as.character(model_a))
    )

    counts <- sub %>%
        count(winner_model, loser_model, name = "wins")
    # Make 2-column data frame of all pairs and counts (BTm uses
    # "win1.adv" style; simpler to use BTabilities with countsToBinomial).
    pair_counts <- countsToBinomial(
        xtabs(wins ~ winner_model + loser_model, data = counts)
    )
    # Use the UNION of player1 and player2 as factor levels so no model
    # gets coerced to NA (countsToBinomial puts each pair in alphabetical
    # order, so player1 never contains the alphabetically-last model and
    # vice versa).
    all_levels <- sort(unique(c(as.character(pair_counts$player1),
                                  as.character(pair_counts$player2))))
    pair_counts$player1 <- factor(pair_counts$player1, levels = all_levels)
    pair_counts$player2 <- factor(pair_counts$player2, levels = all_levels)

    if (!REFERENCE_MODEL %in% all_levels) return(NULL)
    pair_counts$player1 <- relevel(pair_counts$player1, ref = REFERENCE_MODEL)
    pair_counts$player2 <- relevel(pair_counts$player2, ref = REFERENCE_MODEL)

    fit <- tryCatch(
        BTm(cbind(win1, win2), player1, player2, data = pair_counts),
        error = function(e) NULL
    )
    if (is.null(fit)) return(NULL)

    ab <- BTabilities(fit)  # matrix: ability and s.e.
    out <- data.frame(
        model = rownames(ab),
        focal_country = focal_country,
        language = language,
        judge = judge,
        bt_logit = ab[, "ability"],
        se = ab[, "s.e."],
        stringsAsFactors = FALSE
    )
    out$ci_lo <- out$bt_logit - 1.96 * out$se
    out$ci_hi <- out$bt_logit + 1.96 * out$se
    out$n_pairs <- nrow(sub)
    out
}

main <- function() {
    df <- read_all_comparisons()
    cat(sprintf("Loaded %d comparison rows\n", nrow(df)))

    # Strata: per-judge fits AND a pooled "all" fit per (country, language)
    strata <- rbind(
        expand.grid(
            country = unique(df$country),
            language = c("cn", "en"),
            judge = unique(df$judge),
            stringsAsFactors = FALSE
        ),
        expand.grid(
            country = unique(df$country),
            language = c("cn", "en"),
            judge = "all",
            stringsAsFactors = FALSE
        )
    )

    results <- lapply(seq_len(nrow(strata)), function(i) {
        fit_one_stratum(df, strata$country[i], strata$language[i], strata$judge[i])
    })
    results <- bind_rows(results)
    results$focal_country_display <- COUNTRY_DISPLAY[results$focal_country]

    cat(sprintf("Wrote %d BT estimates across %d strata\n",
                nrow(results),
                length(unique(paste(results$focal_country, results$language, results$judge)))))

    # Round for JSON
    results$bt_logit <- round(results$bt_logit, 4)
    results$ci_lo <- round(results$ci_lo, 4)
    results$ci_hi <- round(results$ci_hi, 4)
    results$se <- round(results$se, 4)

    out_path <- "data/cross_model_audit/bt_scores.json"
    write_json(results, out_path, auto_unbox = TRUE, pretty = TRUE)
    cat(sprintf("Written to %s\n", out_path))
}

main()
