Shader "Custom/PoseTelemetryHUD"
{
    Properties
    {
        _BlockPx ("Block Size (px)", Float) = 4
        _OffsetX ("Offset X from left (px)", Float) = 8
        _OffsetY ("Offset Y from top (px)", Float) = 8
        _TextPx ("Text Pixel Size (px)", Float) = 3
    }
    SubShader
    {
        Tags { "Queue"="Overlay+1000" "RenderType"="Overlay" "IgnoreProjector"="True" }
        ZTest Always
        ZWrite Off
        Cull Off
        Blend Off

        Pass
        {
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #pragma target 5.0
            #include "UnityCG.cginc"

            float _BlockPx;
            float _OffsetX;
            float _OffsetY;
            float _TextPx;

            float _VRChatCameraMode;
            float _VRChatMirrorMode;
            uint  _VRChatTimeNetworkMs;

            struct appdata { float4 vertex : POSITION; float2 uv : TEXCOORD0; };
            struct v2f     { float4 pos : SV_Position; };

            v2f vert (appdata v)
            {
                v2f o;
                if (_VRChatCameraMode != 0.0 || _VRChatMirrorMode != 0.0)
                {
                    o.pos = float4(2e6, 2e6, 2e6, 1.0);
                    return o;
                }
                float2 ndc = v.uv * 2.0 - 1.0;
                o.pos = float4(ndc, UNITY_NEAR_CLIP_VALUE, 1.0);
                return o;
            }

            #define ROWS 12
            #define COLS 32
            static const uint MAGIC = 0x5AC3E7A1u;

            // 3x5 pixel font: 3 bits per row, row0 (top) at LSB, left pixel = MSB of each row
            #define G_MINUS 10u
            #define G_DOT   11u
            #define G_SP    12u
            #define G_X     13u
            #define G_Y     14u
            #define G_Z     15u
            #define G_P     16u
            #define G_H     17u
            #define G_R     18u
            static const uint FONT[19] = {
                0x7B6F, 0x74B2, 0x79CF, 0x73CF, 0x13ED, // 0-4
                0x73E7, 0x7BE7, 0x248F, 0x7BEF, 0x73EF, // 5-9
                0x01C0, // -
                0x2000, // .
                0x0000, // space
                0x5AAD, // X
                0x24AD, // Y
                0x788F, // Z
                0x49EF, // P
                0x5BED, // H
                0x5DEF  // R
            };

            #define TXT_CHARS 10
            #define TXT_LINES 6

            // Fixed-width field "L-12345.67": returns glyph index for char ci
            uint glyphAt(float v, uint label, uint ci)
            {
                if (ci == 0u) return label;
                if (ci == 1u) return v < 0.0 ? G_MINUS : G_SP;
                if (ci == 7u) return G_DOT;
                uint s = min((uint)(abs(v) * 100.0 + 0.5), 9999999u);
                if (ci == 8u) return (s / 10u) % 10u;
                if (ci == 9u) return s % 10u;
                uint div = 1u;
                for (uint k = ci; k < 6u; k++) div *= 10u;
                uint ip = s / 100u;
                if (ci < 6u && ip < div) return G_SP; // suppress leading zeros
                return (ip / div) % 10u;
            }

            fixed4 frag (v2f i) : SV_Target
            {
                float2 p = i.pos.xy;
                if (_ProjectionParams.x < 0.0)
                    p.y = _ScreenParams.y - p.y;

                // --- Top-left: binary telemetry grid ---
                float2 g = p - float2(_OffsetX, _OffsetY);
                float2 gridPx = float2(COLS, ROWS) * _BlockPx;

                if (g.x >= 0.0 && g.y >= 0.0 && g.x < gridPx.x && g.y < gridPx.y)
                {
                    uint col = (uint)(g.x / _BlockPx);
                    uint row = (uint)(g.y / _BlockPx);

                    float3 camPos = _WorldSpaceCameraPos;
                    float3 fwd = -UNITY_MATRIX_V[2].xyz;
                    float3 up  =  UNITY_MATRIX_V[1].xyz;

                    uint w[ROWS];
                    w[0]  = MAGIC;
                    w[1]  = _VRChatTimeNetworkMs;
                    w[2]  = asuint(camPos.x);
                    w[3]  = asuint(camPos.y);
                    w[4]  = asuint(camPos.z);
                    w[5]  = asuint(fwd.x);
                    w[6]  = asuint(fwd.y);
                    w[7]  = asuint(fwd.z);
                    w[8]  = asuint(up.x);
                    w[9]  = asuint(up.y);
                    w[10] = asuint(up.z);
                    w[11] = w[0]^w[1]^w[2]^w[3]^w[4]^w[5]
                          ^ w[6]^w[7]^w[8]^w[9]^w[10];

                    uint bit = (w[row] >> (31u - col)) & 1u;
                    return bit ? fixed4(1,1,1,1) : fixed4(0,0,0,1);
                }

                // --- Top-right: human-readable position / attitude HUD ---
                float2 cellPx = float2(4.0, 6.0) * _TextPx; // 3x5 glyph + 1px spacing
                float2 hudPx = float2(TXT_CHARS, TXT_LINES) * cellPx;
                float2 t = p - float2(_ScreenParams.x - _OffsetX - hudPx.x, _OffsetY);

                if (t.x < 0.0 || t.y < 0.0 || t.x >= hudPx.x || t.y >= hudPx.y)
                    clip(-1.0);

                uint ci = (uint)(t.x / cellPx.x);
                uint li = (uint)(t.y / cellPx.y);
                uint rx = (uint)(t.x / _TextPx) % 4u;
                uint ry = (uint)(t.y / _TextPx) % 6u;

                float3 camPos = _WorldSpaceCameraPos;
                float3 right = UNITY_MATRIX_V[0].xyz;
                float3 up    = UNITY_MATRIX_V[1].xyz;
                float3 fwd   = -UNITY_MATRIX_V[2].xyz;

                // Unity euler (deg): P = pitch (X), H = heading/yaw (Y), R = roll (Z)
                float pitch = degrees(asin(clamp(-fwd.y, -1.0, 1.0)));
                float yaw   = degrees(atan2(fwd.x, fwd.z));
                float roll  = degrees(atan2(right.y, up.y));

                float vals[TXT_LINES]   = { camPos.x, camPos.y, camPos.z, pitch, yaw, roll };
                uint  labels[TXT_LINES] = { G_X, G_Y, G_Z, G_P, G_H, G_R };

                uint gph = glyphAt(vals[li], labels[li], ci);
                uint on = 0u;
                if (rx < 3u && ry < 5u)
                    on = (FONT[gph] >> (ry * 3u + (2u - rx))) & 1u;
                return on ? fixed4(1,1,1,1) : fixed4(0,0,0,1);
            }
            ENDCG
        }
    }
    Fallback Off
}
