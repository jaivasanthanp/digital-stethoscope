/*
 * heart_sound_service.c — Custom BLE GATT service
 *
 * Service UUID:        12345678-1234-1234-1234-123456789ABC
 * Characteristic UUID: 12345678-1234-1234-1234-123456789ABD
 *   Properties: NOTIFY
 *   Format: short printable ASCII string, e.g. "Present 95%"
 *   nRF Connect auto-renders ASCII payloads as text in the notification feed.
 *
 * class_id encoding (CirCor 2022, 3-class):
 *   0x00 = Absent
 *   0x01 = Present
 *   0x02 = Unknown
 *   0xFF = Error
 */

#include "heart_sound_service.h"

#include <zephyr/bluetooth/bluetooth.h>
#include <zephyr/bluetooth/gatt.h>
#include <zephyr/logging/log.h>
#include <stdio.h>

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

    BT_GATT_CUD("Heart Sound Classification", BT_GATT_PERM_READ),

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
    ARG_UNUSED(timestamp_ms);

    if (!notify_enabled) {
        return -ENOTCONN;
    }

    static const char *const class_names[] = {"Absent", "Present", "Unknown"};
    const char *name = (class_id < ARRAY_SIZE(class_names))
                           ? class_names[class_id]
                           : "Error";

    /* Printable ASCII payload — e.g. "Present 95%" (no trailing NUL).
     * nRF Connect renders this as text in the notification log line. */
    char pkt[20];
    int len = snprintf(pkt, sizeof(pkt), "%s %u%%", name, confidence);
    if (len < 0) {
        return -EINVAL;
    }
    if (len > (int)sizeof(pkt)) {
        len = sizeof(pkt);
    }

    /* Characteristic value attribute (index 1 in the service table). */
    const struct bt_gatt_attr *attr = &hsc_svc.attrs[1];

    struct bt_gatt_notify_params params = {
        .attr = attr,
        .data = pkt,
        .len  = (uint16_t)len,
    };

    int err = bt_gatt_notify_cb(conn, &params);
    if (err) {
        LOG_ERR("Notify failed: %d", err);
    }
    return err;
}
