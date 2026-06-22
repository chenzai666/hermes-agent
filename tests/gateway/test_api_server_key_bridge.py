from gateway.config import GatewayConfig, Platform


def test_platforms_api_server_top_level_key_is_bridged_to_extra_key():
    cfg = GatewayConfig.from_dict({
        'platforms': {
            'api_server': {
                'enabled': True,
                'key': 'test-secret-key',
                'extra': {'host': '127.0.0.1', 'port': 8642},
            }
        }
    })

    api = cfg.platforms[Platform.API_SERVER]
    assert api.extra['key'] == 'test-secret-key'
    assert api.extra['host'] == '127.0.0.1'
    assert api.extra['port'] == 8642
