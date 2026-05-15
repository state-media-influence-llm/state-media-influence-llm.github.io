#!/usr/bin/env Rscript
# Static reproduction of the cross-model audit BT plot.
#
# Reads data/cross_model_audit/bt_scores_with_language.json and produces a
# horizontal dot plot per focal country, with (model, language) on the y axis
# and odds vs. average on a log-scale x axis. Mirrors the OJS chart on
# cross_model_audit.qmd.
#
# Output: plots/cross_model_audit.pdf
#
# Usage: Rscript scripts/plot_cross_model_audit.R

suppressPackageStartupMessages({
    library(jsonlite)
    library(dplyr)
    library(ggplot2)
    library(scales)
})

script_dir <- function() {
    args <- commandArgs(trailingOnly = FALSE)
    f <- sub("^--file=", "", args[grep("^--file=", args)])
    if (length(f)) dirname(normalizePath(f, mustWork = FALSE)) else getwd()
}
BASE_DIR <- dirname(script_dir())

IN_PATH <- file.path(BASE_DIR, "data", "cross_model_audit", "bt_scores_with_language.json")
OUT_PATH <- file.path(BASE_DIR, "plots", "cross_model_audit.pdf")

bt <- fromJSON(IN_PATH) %>%
    as_tibble() %>%
    filter(judge == "all") %>%
    mutate(odds = exp(bt_logit),
           odds_lo = exp(ci_lo),
           odds_hi = exp(ci_hi),
           language_label = ifelse(language == "cn", "Chinese prompt", "English prompt"))

# Model order: sort by China-CN BT score so the heaviest pro-China model is at top
china_cn <- bt %>%
    filter(focal_country_display == "China", language == "cn") %>%
    arrange(desc(bt_logit))
model_order <- china_cn$model

# Country facet order: China and high-state-influence first
country_order <- c("China", "Russia", "North Korea",
                   "United Kingdom", "United States", "Germany")
country_order <- country_order[country_order %in% unique(bt$focal_country_display)]

bt$model <- factor(bt$model, levels = rev(model_order))
bt$focal_country_display <- factor(bt$focal_country_display, levels = country_order)
bt$language_label <- factor(bt$language_label,
                            levels = c("Chinese prompt", "English prompt"))

dodge_h <- position_nudge(y = 0)
bt$y_pos <- as.numeric(bt$model) +
    ifelse(bt$language == "cn", 0.18, -0.18)

p <- ggplot(bt, aes(x = odds, xmin = odds_lo, xmax = odds_hi,
                    y = y_pos, color = language_label)) +
    geom_vline(xintercept = 1, color = "#999", linewidth = 0.4) +
    geom_errorbar(orientation = "y", width = 0, linewidth = 0.6, alpha = 0.75) +
    geom_point(size = 2.2) +
    facet_wrap(~ focal_country_display, ncol = 1, scales = "free_y",
               strip.position = "top") +
    scale_x_log10(breaks = c(0.05, 0.1, 0.3, 1, 3, 10, 30, 100),
                  labels = function(x) ifelse(x >= 1,
                                              sprintf("%gx", x),
                                              sprintf("1/%d", round(1/x)))) +
    scale_y_continuous(breaks = seq_along(levels(bt$model)),
                       labels = levels(bt$model)) +
    scale_color_manual(values = c("Chinese prompt" = "#dc3545",
                                  "English prompt" = "#0d6efd"),
                       name = NULL) +
    labs(x = "odds vs. average (more favorable to focal country -->)",
         y = NULL,
         title = "Cross-Model Audit (Bradley-Terry, both judges pooled)",
         subtitle = "Odds a model gives a more favorable response to a prompt about that country than other (model, language) entries.") +
    theme_minimal(base_size = 10) +
    theme(legend.position = "top",
          panel.grid.minor.x = element_blank(),
          panel.spacing.y = unit(0.6, "lines"),
          strip.text = element_text(face = "bold", hjust = 0))

dir.create(dirname(OUT_PATH), recursive = TRUE, showWarnings = FALSE)
ggsave(OUT_PATH, plot = p,
       width = 8, height = 2 + length(country_order) * 1.6)
cat(sprintf("Wrote %s\n", OUT_PATH))
