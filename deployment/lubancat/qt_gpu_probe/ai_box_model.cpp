#include "ai_box_model.h"

#include <QSet>

namespace {
const char *const kRoleKeys[] = {
    "trackId", "classId", "className", "color", "x", "y", "width", "height",
    "vx", "vy", "vw", "vh", "frameId", "detectTimestampNs", "missed",
    "predicted", "state", "confidence", "label", "primary"
};
constexpr int kRoleCount = int(sizeof(kRoleKeys) / sizeof(kRoleKeys[0]));

int roleForIndex(int i) { return Qt::UserRole + 1 + i; }
QVariantMap normalized(const QVariantMap &src) {
    QVariantMap out;
    for (int i = 0; i < kRoleCount; ++i)
        out.insert(QString::fromLatin1(kRoleKeys[i]), src.value(QString::fromLatin1(kRoleKeys[i])));
    return out;
}
}

AiBoxModel::AiBoxModel(QObject *parent) : QAbstractListModel(parent) {}

int AiBoxModel::rowCount(const QModelIndex &parent) const {
    return parent.isValid() ? 0 : rows_.size();
}

QVariant AiBoxModel::data(const QModelIndex &index, int role) const {
    if (!index.isValid() || index.row() < 0 || index.row() >= rows_.size()) return {};
    const int roleIndex = role - (Qt::UserRole + 1);
    if (roleIndex < 0 || roleIndex >= kRoleCount) return {};
    return rows_.at(index.row()).value(QString::fromLatin1(kRoleKeys[roleIndex]));
}

QHash<int, QByteArray> AiBoxModel::roleNames() const {
    QHash<int, QByteArray> names;
    for (int i = 0; i < kRoleCount; ++i)
        names.insert(roleForIndex(i), QByteArray(kRoleKeys[i]));
    return names;
}

quint64 AiBoxModel::trackId(const QVariantMap &row) {
    return row.value(QStringLiteral("trackId")).toULongLong();
}

QVector<int> AiBoxModel::changedRoles(const QVariantMap &oldRow, const QVariantMap &newRow) {
    QVector<int> roles;
    for (int i = 0; i < kRoleCount; ++i) {
        const QString key = QString::fromLatin1(kRoleKeys[i]);
        if (oldRow.value(key) != newRow.value(key)) roles.append(roleForIndex(i));
    }
    return roles;
}

void AiBoxModel::setBoxes(const QVariantList &boxes) {
    QVector<QVariantMap> next;
    next.reserve(boxes.size());
    QSet<quint64> nextIds;
    for (int i = 0; i < boxes.size(); ++i) {
        QVariantMap row = normalized(boxes.at(i).toMap());
        const quint64 id = trackId(row);
        if (!id || nextIds.contains(id)) continue;
        nextIds.insert(id);
        row.insert(QStringLiteral("primary"), next.isEmpty());
        next.append(row);
    }

    // Remove rows that really disappeared. Do it from the end so indices
    // remain valid, without ever resetting the whole model.
    QSet<quint64> currentIds;
    for (const QVariantMap &row : rows_) currentIds.insert(trackId(row));
    for (int i = rows_.size() - 1; i >= 0; --i) {
        if (nextIds.contains(trackId(rows_.at(i)))) continue;
        beginRemoveRows(QModelIndex(), i, i);
        rows_.removeAt(i);
        endRemoveRows();
        ++removes_;
    }

    QHash<quint64, int> currentRows;
    for (int i = 0; i < rows_.size(); ++i) currentRows.insert(trackId(rows_.at(i)), i);

    for (int i = 0; i < next.size(); ++i) {
        const quint64 id = trackId(next.at(i));
        auto existing = currentRows.constFind(id);
        if (existing == currentRows.constEnd()) {
            const int row = rows_.size();
            beginInsertRows(QModelIndex(), row, row);
            rows_.append(next.at(i));
            endInsertRows();
            currentRows.insert(id, row);
            ++inserts_;
            continue;
        }
        const int row = existing.value();
        const QVector<int> roles = changedRoles(rows_.at(row), next.at(i));
        if (roles.isEmpty()) continue;
        rows_[row] = next.at(i);
        emit dataChanged(index(row, 0), index(row, 0), roles);
        ++data_changes_;
    }

    // A track that did not appear in a new result is removed by the producer
    // when it expires; never keep a stale row merely to stabilize row count.
}
