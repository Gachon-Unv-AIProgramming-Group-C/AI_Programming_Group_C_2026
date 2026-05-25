import { Module } from '@nestjs/common';
import { McpModule } from './mcp/mcp.module';
import { AppController } from './app.controller';
import { AppService } from './app.service';

@Module({
  imports: [McpModule],
  controllers: [AppController],
  providers: [AppService],
})
export class AppModule {}
