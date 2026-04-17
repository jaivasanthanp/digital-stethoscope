/*
 * heart_sound_service.c — Custom BLE GATT service
 *
 * Service UUID:        12345678-1234-1234-1234-123456789ABC
 * Characteristic UUID: 12345678-1234-1234-1234-123456789ABD
 *   Properties: NOTIFY
 *   Length: 6 bytes
 *   Format: [class_id:u8, confidence:u8, reserved:u8, timestamp:u24_le]
 *
 * class_id encoding:
 *   0x00 = Normal
 *   0x01 = SystolicMurmur
 *   0x02 = DiastolicMurmur
 *   0x03 = S3Gallop
 *   0xFF = NoResult
 */

#include "heart_sound_service.h"

#include <zephyr/bluetooth/bluetooth.h>
#include <zephyr/bluetooth/gatt.h>
#include <zephyr/logging/log.h>

LOG_MODULE_REGISTER(hsc_service, LOG_LEVEL_INF);

/* Custom service UUID: 12345678-1234-1234-1234-123456789ABC */
#define BT_UUID_HSC_SERVICE_VAL \
    BT_UUID_128_ENCODE(0x12345678, 0x1234, 0x1234, 0x1234, 0x123456789ABCULL)

/* Characteristic UUID: 12345678-1234-1234-1234-123456789ABD */
#define BT_UUID_HSC_RESULT_VAL \
    BT_UUID_128_ENCODE(0x12345678, 0x1234, 0x1234, 0x1234, 0x123456789ABDULL)

static struct bt_uuid_128 hsc_service_uuid = BT_UUID_INIT_128(BT_UUID_HSC_SERVICE_VAL);
static struct bt_uuid_128 hsc_result_uuid  = BT_UUID_INIT_128(BT_UUID_HSC_RESULT_VAL);

/* Notification subscriber tracking */
static struct bt_gatt_indicate_params indicate_params;
static bool notify_enabled = false;

static void hsc_ccc_changed(const struct bt_gatt_attr *attr, uint16_t value)
{
    notify_enabled = (value == BT_GATT_CCC_NOTIFY);
    LOG_INF("HSC notifications %s", notify_enabled ? "enabled" : "disabled");
}

/* GATT service definition */
BT_GATT_SERVICE_DEFINE(hsc_svc,
    BT_GATT_PRIMARY_SERVICE(&hsc_service_uuid),

    BT_GATT_CHARACTERISTIC(&hsc_result_uuid.uuid,
                           BT_GATT_CHRC_NOTIFY,
                           BT_GATT_PERM_NONE,
                           NULL, NULL, NULL),

    BT_GATT_CCC(hsc_ccc_changed,
                BT_GATT_PERM_READ | BT_GATT_PERM_WRITE),
);

/* -------------------------------------------------------------------------
 * Public API
 * ------------------------------------------------------------------------- */
void hsc_service_init(void)
{
    LOG_INF("Heart Sound Classification service registered");
}

int hsc_service_notify(struct bt_conn *conn,
                       uint8_t class_id, uint8_t confidence,
                       uint32_t timestamp_ms)
{
    if (!notify_enabled) {
        return -ENOTCONN;
    }

    uint8_t pkt[6];
    pkt[0] = class_id;
    pkt[1] = confidence;
    pkt[2] = 0x00;
    pkt[3] = (uint8_t)(timestamp_ms & 0xFF);
    pkt[4] = (uint8_t)((timestamp_ms >> 8) & 0xFF);
    pkt[5] = (uint8_t)((timestamp_ms >> 16) & 0xFF);

    /* Find the characteristic attribute (index 1 in the service table) */
    const struct bt_gatt_attr *attr = &hsc_svc.attrs[1];

    struct bt_gatt_notify_params params = {
        .attr = attr,
        .data = pkt,
        .len  = sizeof(pkt),
    };

    int err = bt_gatt_notify_cb(conn, &params);
    if (err) {
        LOG_ERR("Notify failed: %d", err);
    }
    return err;
}
