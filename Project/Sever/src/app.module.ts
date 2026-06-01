import { Module } from '@nestjs/common';
import { ConfigModule } from '@nestjs/config';
import { McpModule } from './mcp/mcp.module';
import { AppController } from './app.controller';
import { AppService } from './app.service';

@Module({
  imports: [
    ConfigModule.forRoot({
      isGlobal: true,
      envFilePath: [
        '/home/serize/hallucination-server/.env',
        '.env',
      ],
    }),
    McpModule,
  ],
  controllers: [AppController],
  providers: [AppService],
})
export class AppModule {}
