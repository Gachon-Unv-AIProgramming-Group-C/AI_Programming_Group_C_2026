import { Body, Controller, Post, HttpCode, HttpStatus, Logger } from '@nestjs/common';
import { McpService } from './mcp.service';
import { JsonRpcRequest } from './mcp.types';

@Controller('mcp')
export class McpController {
  private readonly logger = new Logger(McpController.name);

  constructor(private readonly mcpService: McpService) {}

  @Post()
  @HttpCode(HttpStatus.OK)
  handleMcp(@Body() body: JsonRpcRequest | JsonRpcRequest[]) {
    // 배치 요청 처리 (JSON-RPC 2.0 spec)
    if (Array.isArray(body)) {
      this.logger.debug(`배치 요청 ${body.length}개`);
      const responses = body
        .map((req) => this.mcpService.handleRequest(req))
        .filter((res): res is NonNullable<typeof res> => res !== null);
      return responses;
    }

    const result = this.mcpService.handleRequest(body);

    // notification은 응답 없음 (HTTP 204 대신 빈 객체 반환)
    if (result === null) return {};

    return result;
  }
}
