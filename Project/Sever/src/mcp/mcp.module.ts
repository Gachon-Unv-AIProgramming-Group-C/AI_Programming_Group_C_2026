// McpModule은 NestJS의 모듈 단위 구조에서 MCP 관련 클래스를 하나로 묶는 역할을 한다
// controllers: HTTP 요청을 받는 McpController를 등록한다
// providers: 실제 비즈니스 로직을 담당하는 McpService를 의존성 주입 컨테이너에 등록한다
import { Module } from '@nestjs/common';
import { McpController } from './mcp.controller';
import { McpService } from './mcp.service';

@Module({
  controllers: [McpController],  // POST /mcp 요청을 처리할 컨트롤러
  providers: [McpService],       // 할루시네이션 탐지 로직을 제공하는 서비스
})
export class McpModule {}
