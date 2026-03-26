# .Rprofile patch for nf-core/crisprseq CRISPRSEQ_PLOTTER
#
# Bind-mounted as /.Rprofile inside the Singularity container.
# Wraps ggseqlogo and ggsave with tryCatch so plotter.R survives
# edge-case data (empty matrices, fontconfig errors, etc.).
#
# The entire file is wrapped in tryCatch — if anything here fails,
# R prints a warning and continues normally so plotter.R still runs.

tryCatch({
    message("[plotter_patch] .Rprofile loaded — applying patches")

    # ggseqlogo crashes with "subscript out of bounds" when the
    # substitution matrix has zero columns.  Return a blank plot instead.
    .patch_ggseqlogo <- function() {
        if (!requireNamespace("ggseqlogo", quietly = TRUE)) {
            message("[plotter_patch] ggseqlogo not available — skipping patch")
            return(invisible(NULL))
        }
        if (!requireNamespace("ggplot2", quietly = TRUE)) {
            message("[plotter_patch] ggplot2 not available — skipping patch")
            return(invisible(NULL))
        }

        orig <- ggseqlogo::ggseqlogo
        safe_fn <- function(data, ...) {
            tryCatch(
                orig(data, ...),
                error = function(e) {
                    message("[plotter_patch] ggseqlogo error caught: ",
                            conditionMessage(e))
                    ggplot2::ggplot() +
                        ggplot2::theme_void() +
                        ggplot2::annotate("text", x = 0.5, y = 0.5,
                                          label = "no data for logo plot",
                                          size = 4, colour = "grey50")
                }
            )
        }
        utils::assignInNamespace("ggseqlogo", safe_fn, "ggseqlogo")
        message("[plotter_patch] ggseqlogo patched OK")
    }

    # ggsave can fail if fontconfig has no writable cache or the plot
    # object is broken.  Create an empty file so Nextflow finds expected
    # outputs rather than crashing.
    .patch_ggsave <- function() {
        if (!requireNamespace("ggplot2", quietly = TRUE)) {
            message("[plotter_patch] ggplot2 not available — skipping ggsave patch")
            return(invisible(NULL))
        }

        orig <- ggplot2::ggsave
        safe_fn <- function(filename, ...) {
            tryCatch(
                orig(filename, ...),
                error = function(e) {
                    message("[plotter_patch] ggsave error caught for ",
                            filename, ": ", conditionMessage(e))
                    file.create(filename)
                }
            )
        }
        utils::assignInNamespace("ggsave", safe_fn, "ggplot2")
        message("[plotter_patch] ggsave patched OK")
    }

    .patch_ggseqlogo()
    .patch_ggsave()

    message("[plotter_patch] all patches applied successfully")

}, error = function(e) {
    message("[plotter_patch] PATCH FAILED — plotter.R will run unpatched: ",
            conditionMessage(e))
})
