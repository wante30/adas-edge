#pragma once

#include <QAbstractListModel>
#include <QVariantList>
#include <QVariantMap>
#include <QVector>

// Stable, GUI-thread-owned model for the QML AI boxes. Rows are keyed by
// trackId; a detection update emits dataChanged only for changed roles and
// inserts/removes rows only when a track actually appears or disappears.
class AiBoxModel : public QAbstractListModel {
    Q_OBJECT
public:
    enum Role {
        TrackIdRole = Qt::UserRole + 1,
        ClassIdRole,
        ClassNameRole,
        ColorRole,
        XRole,
        YRole,
        WidthRole,
        HeightRole,
        VxRole,
        VyRole,
        VwRole,
        VhRole,
        FrameIdRole,
        DetectTimestampNsRole,
        MissedRole,
        PredictedRole,
        StateRole,
        ConfidenceRole,
        LabelRole,
        PrimaryRole
    };

    explicit AiBoxModel(QObject *parent = nullptr);

    int rowCount(const QModelIndex &parent = QModelIndex()) const override;
    QVariant data(const QModelIndex &index, int role) const override;
    QHash<int, QByteArray> roleNames() const override;

    void setBoxes(const QVariantList &boxes);
    int count() const { return rows_.size(); }

    int resetCount() const { return resets_; }
    int insertCount() const { return inserts_; }
    int removeCount() const { return removes_; }
    int dataChangedCount() const { return data_changes_; }

private:
    static quint64 trackId(const QVariantMap &row);
    static QVector<int> changedRoles(const QVariantMap &oldRow, const QVariantMap &newRow);

    QVector<QVariantMap> rows_;
    int resets_ = 0;
    int inserts_ = 0;
    int removes_ = 0;
    int data_changes_ = 0;
};
