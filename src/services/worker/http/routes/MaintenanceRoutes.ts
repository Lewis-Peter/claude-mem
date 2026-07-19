
import express, { Request, Response } from 'express';
import { z } from 'zod';
import { BaseRouteHandler } from '../BaseRouteHandler.js';
import { validateBody } from '../middleware/validateBody.js';
import { logger } from '../../../../utils/logger.js';
import { SettingsDefaultsManager } from '../../../../shared/SettingsDefaultsManager.js';
import { USER_SETTINGS_PATH } from '../../../../shared/paths.js';
import { runRotationJob, type RotationSummary } from '../../jobs/RotationJob.js';
import type { DatabaseManager } from '../../DatabaseManager.js';

const rotationRunSchema = z.object({
  project: z.string().trim().min(1).optional(),
}).strict();

export class MaintenanceRoutes extends BaseRouteHandler {
  constructor(private dbManager: DatabaseManager) {
    super();
  }

  setupRoutes(app: express.Application): void {
    app.post('/api/maintenance/rotation/run', validateBody(rotationRunSchema), this.handleRunRotation.bind(this));
  }

  private handleRunRotation = this.wrapHandler(async (req: Request, res: Response): Promise<void> => {
    const settings = SettingsDefaultsManager.loadFromFile(USER_SETTINGS_PATH);
    if (settings.CLAUDE_MEM_ROTATION_ENABLED === 'false') {
      res.status(403).json({ error: 'disabled' });
      return;
    }

    const { project } = req.body as z.infer<typeof rotationRunSchema>;

    if (project) {
      const summary = await runRotationJob(this.dbManager, project);
      res.json(summary);
      return;
    }

    // No project given — run sequentially across every known project.
    // Sequential (not Promise.all) so we don't hammer the local LLM with
    // concurrent judgement calls across projects.
    const { projects } = this.dbManager.getSessionStore().getProjectCatalog();
    const summaries: RotationSummary[] = [];

    for (const proj of projects) {
      try {
        const summary = await runRotationJob(this.dbManager, proj);
        summaries.push(summary);
      } catch (error) {
        logger.error('DEDUP', 'Rotation job failed for project; continuing with remaining projects', { project: proj }, error as Error);
      }
    }

    res.json(summaries);
  });
}
