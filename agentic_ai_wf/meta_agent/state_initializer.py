"""Initialize meta state from analysis directory and turn metadata."""

import logging
import os
from typing import Optional, Tuple, Any

logger = logging.getLogger(__name__)


async def discover_files_from_analysis_dir(
    analysis_transcriptome_dir: Optional[str],
    turn_id: Optional[int],
    workflow_logger: Any,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Discover prioritized_genes_path and pathway_consolidation_path from:
    1. Filename patterns in analysis_transcriptome_dir
    2. ConversationTurn metadata if turn_id provided
    """
    prioritized: Optional[str] = None
    pathways: Optional[str] = None

    if analysis_transcriptome_dir and os.path.exists(analysis_transcriptome_dir):
        try:
            from django.conf import settings

            for filename in os.listdir(analysis_transcriptome_dir):
                file_path = os.path.join(analysis_transcriptome_dir, filename)
                if not os.path.isfile(file_path):
                    continue
                fn_lower = filename.lower()

                has_degs = any(p in fn_lower for p in ["degs", "deg"])
                has_prioritized = any(p in fn_lower for p in ["prioritized", "prioritised", "prioritize"])
                if has_degs and (has_prioritized or not prioritized):
                    if not prioritized or (has_prioritized and "prioritized" not in (prioritized or "").lower() and "prioritised" not in (prioritized or "").lower()):
                        prioritized = os.path.abspath(file_path)
                        if workflow_logger:
                            await workflow_logger.info(
                                agent_name="Meta Agent",
                                message=f"Found DEGs file from filename pattern: {filename}",
                                step="initialization",
                            )
                        logger.info("Found prioritized_genes_path from pattern: %s", filename)
                        continue

                has_pathways = any(p in fn_lower for p in ["pathways", "pathway"])
                has_consolidated = "consolidated" in fn_lower
                if has_pathways and (has_consolidated or not pathways):
                    if not pathways or (has_consolidated and "consolidated" not in (pathways or "").lower()):
                        pathways = os.path.abspath(file_path)
                        if workflow_logger:
                            await workflow_logger.info(
                                agent_name="Meta Agent",
                                message=f"Found Pathways file from filename pattern: {filename}",
                                step="initialization",
                            )
                        logger.info("Found pathway_consolidation_path from pattern: %s", filename)

            if turn_id and (not prioritized or not pathways):
                prioritized, pathways = await _load_from_turn_metadata(
                    turn_id, prioritized, pathways, workflow_logger
                )
        except Exception as e:
            logger.warning("Failed to scan analysis directory: %s", e)
            if workflow_logger:
                await workflow_logger.warning(
                    agent_name="Meta Agent",
                    message=f"Could not scan directory for file patterns: {e}",
                    step="initialization",
                )

    return prioritized, pathways


async def _load_from_turn_metadata(
    turn_id: int,
    prioritized: Optional[str],
    pathways: Optional[str],
    workflow_logger: Any,
) -> Tuple[Optional[str], Optional[str]]:
    """Load file paths from ConversationTurn metadata."""
    try:
        from django.conf import settings
        from agenticaib.db_pool import sync_to_async_with_cleanup
        from analysisapp.models import ConversationTurn

        def get_metadata(tid: int):
            try:
                turn = ConversationTurn.objects.get(id=tid)
                return turn.metadata or {}
            except ConversationTurn.DoesNotExist:
                return {}

        turn_metadata = await sync_to_async_with_cleanup(get_metadata)(turn_id)
        file_mapping = turn_metadata.get("file_mapping", {})

        if not prioritized and "genes_file" in file_mapping:
            info = file_mapping["genes_file"]
            path = info.get("path") if isinstance(info, dict) else str(info)
            if path:
                if not os.path.isabs(path):
                    path = os.path.join(settings.MEDIA_ROOT, path)
                if os.path.exists(path):
                    prioritized = os.path.abspath(path)
                    if workflow_logger:
                        await workflow_logger.info(
                            agent_name="Meta Agent",
                            message=f"Initialized prioritized_genes_path from metadata: {os.path.basename(prioritized)}",
                            step="initialization",
                        )
                    logger.info("Initialized prioritized_genes_path from metadata: %s", prioritized)

        if not pathways and "pathways_file" in file_mapping:
            info = file_mapping["pathways_file"]
            path = info.get("path") if isinstance(info, dict) else str(info)
            if path:
                if not os.path.isabs(path):
                    path = os.path.join(settings.MEDIA_ROOT, path)
                if os.path.exists(path):
                    pathways = os.path.abspath(path)
                    if workflow_logger:
                        await workflow_logger.info(
                            agent_name="Meta Agent",
                            message=f"Initialized pathway_consolidation_path from metadata: {os.path.basename(pathways)}",
                            step="initialization",
                        )
                    logger.info("Initialized pathway_consolidation_path from metadata: %s", pathways)
    except Exception as e:
        logger.warning("Failed to load file_mapping from ConversationTurn metadata: %s", e)
        if workflow_logger:
            await workflow_logger.warning(
                agent_name="Meta Agent",
                message=f"Could not initialize file paths from metadata: {e}",
                step="initialization",
            )
    return prioritized, pathways
