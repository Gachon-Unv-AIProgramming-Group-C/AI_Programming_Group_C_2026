import { NestFactory } from '@nestjs/core';
import { AppModule } from './app.module';
import { McpService } from './mcp/mcp.service';
import * as readline from 'readline';
import { LoggerService } from '@nestjs/common';

class StderrLogger implements LoggerService {
  log(message: any, ...optionalParams: any[]) {
    console.error(`[LOG] ${message}`, ...optionalParams);
  }
  error(message: any, ...optionalParams: any[]) {
    console.error(`[ERROR] ${message}`, ...optionalParams);
  }
  warn(message: any, ...optionalParams: any[]) {
    console.error(`[WARN] ${message}`, ...optionalParams);
  }
  debug(message: any, ...optionalParams: any[]) {
    console.error(`[DEBUG] ${message}`, ...optionalParams);
  }
  verbose(message: any, ...optionalParams: any[]) {
    console.error(`[VERBOSE] ${message}`, ...optionalParams);
  }
}

async function bootstrap() {
  const isStdio = process.argv.includes('--stdio');

  if (isStdio) {
    // stdio mode: Communicate via MCP JSON-RPC on stdin/stdout instead of running an HTTP server
    const app = await NestFactory.createApplicationContext(AppModule, {
      logger: new StderrLogger(),
    });

    const mcpService = app.get(McpService);

    // Register callback to send messages to the MCP client
    mcpService.registerWriteCallback((msg) => {
      process.stdout.write(JSON.stringify(msg) + '\n');
    });

    const rl = readline.createInterface({
      input: process.stdin,
      output: process.stderr,
      terminal: false,
    });

    rl.on('line', async (line) => {
      const trimmed = line.trim();
      if (!trimmed) return;

      try {
        const req = JSON.parse(trimmed);
        const res = await mcpService.handleRequest(req);
        if (res) {
          process.stdout.write(JSON.stringify(res) + '\n');
        }
      } catch (err) {
        console.error('Error handling line:', err);
        process.stdout.write(
          JSON.stringify({
            jsonrpc: '2.0',
            id: null,
            error: {
              code: -32700,
              message: `Parse error: ${err.message}`,
            },
          }) + '\n'
        );
      }
    });

    process.on('SIGINT', async () => {
      await app.close();
      process.exit(0);
    });
    process.on('SIGTERM', async () => {
      await app.close();
      process.exit(0);
    });
  } else {
    // HTTP mode
    const app = await NestFactory.create(AppModule);
    app.enableCors();

    const port = process.env.PORT ?? 8000;
    await app.listen(port, '0.0.0.0');
    console.log(`Server running on: http://localhost:${port}`);
    console.log(`MCP endpoint: http://localhost:${port}/mcp`);
  }
}
bootstrap();

