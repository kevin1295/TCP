#!/usr/bin/env python3
"""
通用的 CARLA 地图切换脚本
支持命令行参数指定：目标地图、CARLA 主机、端口
示例用法：
    python switch_map.py Town01                # 切换到 Town01，默认 localhost:2000
    python switch_map.py Town02 192.168.1.100  # 切换到 Town02，连接 192.168.1.100:2000
    python switch_map.py Town03 localhost 2001 # 切换到 Town03，连接 localhost:2001
"""

import carla
import time
import argparse
import sys

def switch_carla_map(target_town: str, carla_host: str = "localhost", carla_port: int = 2000, wait_time: int = 5):
    """
    切换 CARLA 服务器的地图
    :param target_town: 目标地图名（如 Town01、Town02、Town10）
    :param carla_host: CARLA 服务器主机地址
    :param carla_port: CARLA 服务器端口
    :param wait_time: 地图加载等待时间（秒）
    """
    # 1. 构建完整的地图路径（CARLA 要求的格式）
    target_map_path = f"Carla/Maps/{target_town}"
    
    try:
        # 2. 连接 CARLA 服务器
        print(f"🔌 正在连接 CARLA 服务器: {carla_host}:{carla_port}")
        client = carla.Client(carla_host, carla_port)
        client.set_timeout(15.0)  # 超时时间15秒（足够连接远程服务器）

        # 3. 获取当前地图信息
        current_world = client.get_world()
        current_map = current_world.get_map()
        current_map_path = current_map.name
        current_map_name = current_map_path.split("/")[-1]  # 提取纯地图名（如 Town01）
        print(f"📌 当前加载的地图: {current_map_name} (完整路径: {current_map_path})")

        # 4. 检查是否已加载目标地图
        if current_map_path == target_map_path:
            print(f"✅ 无需切换！当前已加载目标地图: {target_town}")
            return True

        # 5. 切换地图
        print(f"🔄 正在切换到目标地图: {target_town} (完整路径: {target_map_path})")
        new_world = client.load_world(target_map_path)
        
        # 6. 等待地图加载完成
        print(f"⌛ 等待 {wait_time} 秒，确保地图加载完毕...")
        time.sleep(wait_time)

        # 7. 验证切换结果
        new_map = new_world.get_map()
        new_map_name = new_map.name.split("/")[-1]
        if new_map.name == target_map_path:
            print(f"✅ 地图切换成功！当前地图: {new_map_name}")
            return True
        else:
            print(f"❌ 地图切换失败！预期: {target_town}, 实际: {new_map_name}")
            return False

    except carla.exceptions.CarlaConnectionError:
        print(f"\n❌ 连接失败！请检查：")
        print(f"  1. CARLA 服务器是否在 {carla_host}:{carla_port} 运行")
        print(f"  2. 防火墙/网络是否允许访问该端口")
        return False
    except Exception as e:
        print(f"\n❌ 切换地图时发生未知错误: {type(e).__name__}: {e}")
        return False

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="CARLA 地图切换工具")
    parser.add_argument("town", help="目标地图名（如 Town01、Town02、Town10）")
    parser.add_argument("host", nargs="?", default="localhost", help="CARLA 服务器主机（默认: localhost）")
    parser.add_argument("port", nargs="?", type=int, default=2000, help="CARLA 服务器端口（默认: 2000）")
    parser.add_argument("--wait", type=int, default=5, help="地图加载等待时间（秒，默认: 5）")
    
    args = parser.parse_args()

    # 执行地图切换
    success = switch_carla_map(
        target_town=args.town,
        carla_host=args.host,
        carla_port=args.port,
        wait_time=args.wait
    )

    # 根据结果返回退出码
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()